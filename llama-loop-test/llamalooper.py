import discord
from discord.ext import commands
import nest_asyncio
import asyncio
import time
import ctypes
import sys
import pathlib
from ctypes import c_int, c_float, c_char_p, c_void_p, c_bool, POINTER, Structure, Array, c_uint8, c_size_t

# Load the llama library
def _load_shared_library(lib_base_name):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".dylib"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}"
    ]

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path))
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(f"Shared library with base name '{lib_base_name}' not found")

_lib_base_name = "llama"
_lib = _load_shared_library(_lib_base_name)

# Define the ctypes types used by the Llama library
llama_context_p = c_void_p
llama_token = c_int
llama_token_p = POINTER(llama_token)

class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),  # token id
        ("p", c_float),  # probability of the token
        ("plog", c_float),  # log probability of the token
    ]

llama_token_data_p = POINTER(llama_token_data)
llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)

class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),  # text context
        ("n_parts", c_int),  # -1 for default
        ("seed", c_int),  # RNG seed, 0 for random
        ("f16_kv", c_bool),  # use fp16 for KV cache
        ("logits_all", c_bool),  # the llama_eval() call computes all logits, not just the last one
        ("vocab_only", c_bool),  # only load the vocabulary, no weights
        ("use_mlock", c_bool),  # force system to keep model in RAM
        ("embedding", c_bool),  # embedding mode only
        # called with a progress value between 0 and 1, pass NULL to disable
        ("progress_callback", llama_progress_callback),
        # context pointer passed to the progress callback
        ("progress_callback_user_data", c_void_p),
   class llama_context(Structure):
    _fields_ = [
        ("handle", llama_context_p),
        ("params", llama_context_params),
    ]

# Define Llama library functions
_lib.llama_version.restype = c_char_p
_lib.llama_init.argtypes = [c_char_p, POINTER(llama_context_params)]
_lib.llama_init.restype = llama_context_p
_lib.llama_eval.argtypes = [llama_context_p, POINTER(llama_token), c_int, POINTER(llama_token_data)]
_lib.llama_eval.restype = c_void_p
_lib.llama_destroy.argtypes = [llama_context_p]
_lib.llama_destroy.restype = None

# Load the model and create a context
_model_path = pathlib.Path(__file__).parent.resolve() / "model"
_model_path_str = str(_model_path)
_context_params = llama_context_params(n_ctx=1024, n_parts=-1, seed=0, f16_kv=False,
                                       logits_all=False, vocab_only=False, use_mlock=False,
                                       embedding=False, progress_callback=None,
                                       progress_callback_user_data=None)
_context = llama_context(handle=None, params=_context_params)
_lib.llama_init(_model_path_str.encode(), byref(_context.params), byref(_context.handle))

# Define a function for generating Llama tokens from text
def generate_tokens(text):
    text_bytes = text.encode()
    text_len = len(text_bytes)
    token_array = (llama_token * text_len)()
    for i in range(text_len):
        token_array[i] = text_bytes[i]
    token_ptr = llama_token_p(token_array)
    token_data_array = (llama_token_data * text_len)()
    _lib.llama_eval(_context.handle, token_ptr, text_len, token_data_array)
    return token_data_array

# Define the Discord bot and command
bot = commands.Bot(command_prefix='$')

@bot.command()
async def llama(ctx, *, message: str):
    start_time = time.monotonic()
    tokens = generate_tokens(message)
    response = "".join(chr(tokens[i].id) for i in range(len(tokens)))
    await ctx.send(response)
    end_time = time.monotonic()
    print(f"Message: {message}, Response: {response}, Time: {end_time - start_time:.3f}")

# Run the Discord bot
nest_asyncio.apply()
bot.run("TOKEN")
