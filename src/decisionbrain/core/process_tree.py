"""Cross-platform ownership and termination for subprocess trees."""

from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import subprocess
from ctypes import Structure, byref, c_size_t, c_ulong, c_ulonglong, c_void_p, sizeof
from pathlib import Path
from typing import Any


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _JobObjectBasicLimitInformation(Structure):
    _fields_ = [
        ("per_process_user_time_limit", c_ulonglong),
        ("per_job_user_time_limit", c_ulonglong),
        ("limit_flags", c_ulong),
        ("minimum_working_set_size", c_size_t),
        ("maximum_working_set_size", c_size_t),
        ("active_process_limit", c_ulong),
        ("affinity", c_size_t),
        ("priority_class", c_ulong),
        ("scheduling_class", c_ulong),
    ]


class _IoCounters(Structure):
    _fields_ = [
        ("read_operation_count", c_ulonglong),
        ("write_operation_count", c_ulonglong),
        ("other_operation_count", c_ulonglong),
        ("read_transfer_count", c_ulonglong),
        ("write_transfer_count", c_ulonglong),
        ("other_transfer_count", c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(Structure):
    _fields_ = [
        ("basic_limit_information", _JobObjectBasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", c_size_t),
        ("job_memory_limit", c_size_t),
        ("peak_process_memory_used", c_size_t),
        ("peak_job_memory_used", c_size_t),
    ]


class _WindowsJob:
    def __init__(self, pid: int) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (c_void_p, c_void_p)
        kernel32.CreateJobObjectW.restype = c_void_p
        kernel32.SetInformationJobObject.argtypes = (c_void_p, c_ulong, c_void_p, c_ulong)
        kernel32.SetInformationJobObject.restype = c_ulong
        kernel32.OpenProcess.argtypes = (c_ulong, c_ulong, c_ulong)
        kernel32.OpenProcess.restype = c_void_p
        kernel32.AssignProcessToJobObject.argtypes = (c_void_p, c_void_p)
        kernel32.AssignProcessToJobObject.restype = c_ulong
        kernel32.TerminateJobObject.argtypes = (c_void_p, c_ulong)
        kernel32.TerminateJobObject.restype = c_ulong
        kernel32.CloseHandle.argtypes = (c_void_p,)
        kernel32.CloseHandle.restype = c_ulong

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise OSError("CreateJobObjectW failed")
        process_handle = None
        try:
            limits = _JobObjectExtendedLimitInformation()
            limits.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                job,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                byref(limits),
                sizeof(limits),
            ):
                raise OSError("SetInformationJobObject failed")
            process_handle = kernel32.OpenProcess(
                _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
                False,
                pid,
            )
            if not process_handle:
                raise OSError(f"OpenProcess failed for PID {pid}")
            if not kernel32.AssignProcessToJobObject(job, process_handle):
                raise OSError(f"AssignProcessToJobObject failed for PID {pid}")
        except BaseException:
            kernel32.CloseHandle(job)
            raise
        finally:
            if process_handle:
                kernel32.CloseHandle(process_handle)
        self._kernel32 = kernel32
        self._handle = job

    def terminate(self) -> None:
        if self._handle:
            self._kernel32.TerminateJobObject(self._handle, 1)
            self.close()

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class ProcessTree:
    """Own a shell process and every descendant it creates."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        windows_job: _WindowsJob | None = None,
    ) -> None:
        self.process = process
        self._windows_job = windows_job
        self._closed = False

    @classmethod
    async def create_shell(
        cls,
        command: str,
        *,
        cwd: Path,
        stdout: int,
        stderr: int,
        env: dict[str, str] | None = None,
    ) -> ProcessTree:
        platform_options: dict[str, Any]
        if os.name == "nt":
            platform_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            platform_options = {"start_new_session": True}

        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            env=env,
            **platform_options,
        )
        windows_job = None
        if os.name == "nt":
            try:
                windows_job = _WindowsJob(process.pid)
            except BaseException:
                process.kill()
                await process.wait()
                raise
        return cls(process, windows_job=windows_job)

    @classmethod
    async def create_exec(
        cls,
        program: str,
        *args: str,
        cwd: Path,
        stdout: int,
        stderr: int,
        env: dict[str, str] | None = None,
    ) -> ProcessTree:
        platform_options: dict[str, Any]
        if os.name == "nt":
            platform_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            platform_options = {"start_new_session": True}

        process = await asyncio.create_subprocess_exec(
            program,
            *args,
            cwd=str(cwd),
            stdout=stdout,
            stderr=stderr,
            env=env,
            **platform_options,
        )
        windows_job = None
        if os.name == "nt":
            try:
                windows_job = _WindowsJob(process.pid)
            except BaseException:
                process.kill()
                await process.wait()
                raise
        return cls(process, windows_job=windows_job)

    def terminate(self) -> None:
        if self._closed:
            return
        if self._windows_job is not None:
            self._windows_job.terminate()
            return
        self._signal_process_group(signal.SIGTERM)

    def kill(self) -> None:
        if self._closed:
            return
        if self._windows_job is not None:
            self._windows_job.terminate()
            return
        self._signal_process_group(signal.SIGKILL)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._windows_job is not None:
            self._windows_job.close()
            return
        # A shell may exit after starting a background child with redirected output.
        self._signal_process_group(signal.SIGKILL)

    def _signal_process_group(self, sig: signal.Signals) -> None:
        try:
            os.killpg(self.process.pid, sig)
        except ProcessLookupError:
            pass
