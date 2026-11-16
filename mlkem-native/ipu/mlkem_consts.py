"""ML-KEM constants (q, zetas) as used by mlkem-native, and helpers."""
import numpy as np

Q = 3329
N = 256
MONT_R = 1 << 16
RINV = pow(MONT_R, -1, Q)          # 169
MAGIC = 1.5 * 2 ** 23              # fl(x + MAGIC) - MAGIC = round-half-even(x) for |x| < 2^22

# mlkem-native's zeta table (Montgomery form, signed canonical), copied from mlkem/src/zetas.inc
ZETAS_MONT = [
    -1044, -758, -359, -1517, 1493, 1422, 287, 202, -171, 622, 1577,
    182, 962, -1202, -1474, 1468, 573, -1325, 264, 383, -829, 1458,
    -1602, -130, -681, 1017, 732, 608, -1542, 411, -205, -1571, 1223,
    652, -552, 1015, -1293, 1491, -282, -1544, 516, -8, -320, -666,
    -1618, -1162, 126, 1469, -853, -90, -271, 830, 107, -1421, -247,
    -951, -398, 961, -1508, -725, 448, -1065, 677, -1275, -1103, 430,
    555, 843, -1251, 871, 1550, 105, 422, 587, 177, -235, -291,
    -460, 1574, 1653, -246, 778, 1159, -147, -777, 1483, -602, 1119,
    -1590, 644, -872, 349, 418, 329, -156, -75, 817, 1097, 603,
    610, 1322, -1285, -1465, 384, -1215, -136, 1218, -1335, -874, 220,
    -1187, -1659, -1185, -1530, -1278, 794, -1510, -854, -870, 478, -108,
    -308, 996, 991, 958, -1460, 1522, 1628,
]


def centered(x):
    x = np.asarray(x, dtype=np.int64) % Q
    return np.where(x > Q // 2, x - Q, x)


# plain (non-Montgomery) twiddles: zeta_mont * R^-1 mod q, centered
ZETAS = [int(centered(z * RINV)) for z in ZETAS_MONT]


def f32(x):
    return np.float32(x)


def ntt_ref(a):
    """Exact forward NTT mod q (mlkem-native order, output mod q in [0,q))."""
    r = [int(x) % Q for x in a]
    k = 1
    ln = 128
    while ln >= 2:
        for start in range(0, 256, 2 * ln):
            z = ZETAS[k] % Q
            k += 1
            for j in range(start, start + ln):
                t = z * r[j + ln] % Q
                r[j + ln] = (r[j] - t) % Q
                r[j] = (r[j] + t) % Q
        ln >>= 1
    return np.array(r, dtype=np.int64)


def intt_ref(a):
    """mlkem-native invntt_tomont mod q: inverse NTT times 2^16 (Montgomery factor), in [0,q)."""
    r = [int(x) % Q for x in a]
    ln = 2
    k = 127
    while ln <= 128:
        for start in range(0, 256, 2 * ln):
            z = ZETAS[k] % Q
            k -= 1
            for j in range(start, start + ln):
                t = r[j]
                r[j] = (t + r[j + ln]) % Q
                r[j + ln] = z * (r[j + ln] - t) % Q
        ln <<= 1
    f = pow(128, -1, Q) * (1 << 16) % Q
    return np.array([x * f % Q for x in r], dtype=np.int64)
