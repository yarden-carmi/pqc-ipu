ML-KEM on the LEO-NG IPU

One folder per implementation:
  mlkem-native/   ML-KEM-768, unprotected, reference mlkem-native: entirely on the IPU in NX8, bit-exact (see mlkem-native/README.txt)

The IPU emulator is shared by all implementations: emu/ is a git submodule (yarden-carmi/ipu-emulator,
branch pqc-ipu-pin, commit 4ebd3918). Get it with:
  git submodule update --init
