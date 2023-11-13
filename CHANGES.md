Version 1.8.1
-------------
- Add CHANGES.md
- Add boolean mask operations (and, or, xor, invert).
- When setting pixels with None, do not expand the coverage mask.

Version 1.8.0
-------------
- Add bit-packed masks for efficient representation of boolean maps in memory.
- Change to RICE compression for integer maps in FITS.
- Use section API in astropy for faster tile reading.
- Add support for reshaping very large compressed images persisted in FITS.
