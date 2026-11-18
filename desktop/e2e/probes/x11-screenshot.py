"""Capture the disposable GitHub Xvfb display using already-installed GTK."""
import ctypes as c
import os
from pathlib import Path
import sys


def bind(library, name, result, *args):
    fn = getattr(library, name)
    fn.restype, fn.argtypes = result, args
    return fn


if sys.platform != "linux" or os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get("DISPLAY"):
    raise RuntimeError("The autocomplete screenshot requires the Linux CI display.")

target = Path(sys.argv[1])
if not target.is_absolute() or target.exists() or target.suffix != ".png":
    raise ValueError("Expected an unused absolute PNG output path.")

gtk = c.CDLL("libgtk-3.so.0")
gdk = c.CDLL("libgdk-3.so.0")
pixbuf = c.CDLL("libgdk_pixbuf-2.0.so.0")
gobject = c.CDLL("libgobject-2.0.so.0")
ptr = c.c_void_p
integer = c.c_int
if not bind(gtk, "gtk_init_check", integer, ptr, ptr)(None, None):
    raise RuntimeError("GTK could not open the CI display.")
window = bind(gdk, "gdk_get_default_root_window", ptr)()
if not window:
    raise RuntimeError("There is no root window on the CI display.")
width = bind(gdk, "gdk_window_get_width", integer, ptr)(window)
height = bind(gdk, "gdk_window_get_height", integer, ptr)(window)
if (width, height) != (1280, 1024):
    raise RuntimeError(f"Unexpected CI display size: {width}x{height}")
image = bind(gdk, "gdk_pixbuf_get_from_window", ptr, ptr, integer, integer, integer, integer)(
    window, 0, 0, width, height
)
if not image:
    raise RuntimeError("Could not capture the CI display.")
try:
    save = bind(pixbuf, "gdk_pixbuf_savev", integer, ptr, c.c_char_p, c.c_char_p, ptr, ptr, ptr)
    if not save(image, os.fsencode(target), b"png", None, None, None):
        raise RuntimeError(f"Could not save {target.name}.")
finally:
    bind(gobject, "g_object_unref", None, ptr)(image)
