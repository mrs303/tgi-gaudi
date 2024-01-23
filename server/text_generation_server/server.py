import asyncio
import os
import sys
import torch

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models import Model, get_model
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor

from .profiler import Profiler

import time
import shelve
import signal
import concurrent.futures
import threading
import torch.profiler

class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        wait_steps = 390
        warmup_steps = 3
        active_steps = 3
        repeat = 1
        trace_dir = "."
        worker_name = "TextGenerationService"

        dummy = lambda : 0

        if int(os.getenv("RANK", "0")) == 0:
            self.torch_profiler = torch.profiler.profile(
                        activities=(torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU),
                        schedule=torch.profiler.schedule(wait=wait_steps, warmup=warmup_steps, active=active_steps, repeat=repeat),
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, worker_name=worker_name, use_gzip=True),
                        record_shapes=False,
                        with_stack=True)
            self.torch_profiler.start()

            self.step_no = 1
            def _step():
                self.torch_profiler.step()
                out = self.step_no
                self.step_no += 1
                return out
            self._step = _step

            def _profiler_stop():
                logger.info("Stopping profiler...")
                self.torch_profiler.stop()
                logger.info("Profiler stopped")
                self._step = dummy
                self.profiler_stop = dummy

            self.profiler_stop = _profiler_stop
        else:
            self._step = dummy
            self.profiler_stop = dummy

        self.profiler = Profiler()
        with self.profiler.record_event("external", "init"):
            self.cache = cache
            self.model = model
            self.server_urls = server_urls
            # For some reason, inference_mode does not work well with GLOO which we use on CPU
            # TODO: The inferecemode set messes up the autograd op dispatch. And results in aten::matmul
            # op not optimized issue. Will investigate further.
            # if model.device.type == "hpu":
            # Force inference mode for the lifetime of TextGenerationService
            # self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "hpu":
            torch.zeros((2, 2)).to("hpu")
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        with self.profiler.record_event("external", "clear_cache"):
            if request.HasField("id"):
                self.cache.delete(request.id)
            else:
                self.cache.clear()
            return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        batch = self.cache.pop(request.batch_id)
        with self.profiler.record_event("external",
                                        "filter_batch",
                                        {"batch_id": request.batch_id, "request_ids": [id for id in request.request_ids]},
                                        {"util": len(batch.requests)}):
            if batch is None:
                raise ValueError(f"Batch ID {request.batch_id} not found in cache.")
            filtered_batch = batch.filter(request.request_ids)
            self.cache.set(filtered_batch)

            return generate_pb2.FilterBatchResponse(batch=filtered_batch.to_pb())

    async def Warmup(self, request, context):
        with self.profiler.record_event("external", "warmup"):
            # batch = self.model.batch_type.from_pb(
            #     request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            # )
            # max_supported_total_tokens = self.model.warmup(batch)

            # return generate_pb2.WarmupResponse(
            #     max_supported_total_tokens=max_supported_total_tokens
            # )
            logger.warning("Warmup is not enabled on HPU.")
            return generate_pb2.WarmupResponse()

    async def Prefill(self, request, context):
        step_no = self._step()
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.dtype, self.model.device, self.model.is_optimized_for_gaudi
        )
        with self.profiler.record_event("external", "prefill#{}".format(step_no), {"batch_size": batch.input_ids.size(0)}):

            with self.profiler.record_event("internal", "generate_token"):
                generations, next_batch = self.model.generate_token(batch)
            self.cache.set(next_batch)

            return generate_pb2.PrefillResponse(
                generations=[generation.to_pb() for generation in generations],
                batch=next_batch.to_pb() if next_batch else None,
            )

    async def Decode(self, request, context):
        step_no = self._step()
        batch0 = self.cache.cache[request.batches[0].id]
        with self.profiler.record_event("external",
                                        "decode#{}".format(step_no),
                                        {"request_batches": [batch.id for batch in request.batches], "batch_size": batch0.input_ids.size(0)},
                                        {"util": len(batch0.requests)}):
            if len(request.batches) == 0:
                raise ValueError("Must provide at least one batch")

            batches = []
            for batch_pb in request.batches:
                batch = self.cache.pop(batch_pb.id)
                if batch is None:
                    raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
                batches.append(batch)

            if len(batches) == 0:
                raise ValueError("All batches are empty")

            if len(batches) > 1:
                with self.profiler.record_event("internal", "concatenate"):
                    batch = self.model.batch_type.concatenate(batches, self.model.tokenizer.pad_token_id)
            else:
                batch = batches[0]

            with self.profiler.record_event("internal", "generate_token"):
                generations, next_batch = self.model.generate_token(batch)
            self.cache.set(next_batch)

            return generate_pb2.DecodeResponse(
                generations=[generation.to_pb() for generation in generations],
                batch=next_batch.to_pb() if next_batch else None,
            )


def serve(
    model_id: str,
    revision: Optional[str],
    dtype: Optional[str],
    uds_path: Path,
    sharded: bool,
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level="INFO",
        serialize=False,
        backtrace=True,
        diagnose=False,
    )

    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        dtype: Optional[str] = None,
        sharded: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        logger.info("Server:server_inner: sharded ={}".format(sharded))

        if sharded:
            rank = int(os.environ["RANK"])
            logger.info("Server:server_inner: rank ={}".format(rank))
            server_urls = [
                unix_socket_template.format(uds_path, rank) for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        logger.info("Server:server_inner: data type = {}, local_url = {}".format(dtype, local_url))
        if dtype == "bfloat16" or None:
            data_type = torch.bfloat16
        else:
            data_type = torch.float
        if revision == "None":
            revision = None
        try:
            model = get_model(model_id, revision=revision, dtype=data_type)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        current_handler = signal.getsignal(signal.SIGTERM)

        tgi_service = TextGenerationService(model, Cache(), server_urls)

        def handler(sig, frame):
            tgi_service.profiler_stop()
            if callable(current_handler):
                current_handler(sig, frame)

        signal.signal(signal.SIGTERM, handler)
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            tgi_service, server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)
        finally:
            if hasattr(model,'finish_quantization_measurements'):
                model.finish_quantization_measurements()

    logger.info(
        "Starting Server : model_id= {}, revision = {}  dtype = {}  sharded = {} ".format(
            model_id, revision, dtype, sharded
        )
    )
    asyncio.run(serve_inner(model_id, revision, dtype, sharded))
