// Windows only, linked into proxichorizo-cl.exe: OpenCL.dll comes with the AMD GPU driver and is
// delay-loaded (/DELAYLOAD:OpenCL.dll), so --help, --check-pool and the stats tools start without it.
// When the first OpenCL call cannot load it, explain what is missing instead of crashing.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <delayimp.h>
#include <stdio.h>

static FARPROC WINAPI px_opencl_missing(unsigned notify, PDelayLoadInfo info) {
    if (notify == dliFailLoadLib || notify == dliFailGetProc) {
        fprintf(stderr, "\nOpenCL runtime not available (%s%s%s). Install or update the AMD Adrenalin GPU driver,\n"
                        "which provides OpenCL.dll, then start ProxiChorizo again. For NVIDIA GPUs use the CUDA package.\n",
                info && info->szDll ? info->szDll : "OpenCL.dll",
                notify == dliFailGetProc ? ", missing function " : "",
                notify == dliFailGetProc && info && info->dlp.fImportByName ? info->dlp.szProcName : "");
        fflush(stderr);
        ExitProcess(3);
    }
    return nullptr;
}
extern "C" const PfnDliHook __pfnDliFailureHook2 = px_opencl_missing;
#endif
