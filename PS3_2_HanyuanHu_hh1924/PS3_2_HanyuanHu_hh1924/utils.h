#pragma once

#define MATRIX_GET(ptr, x, y, Y) ptr[(x) * (Y) + (y)]

#define FOR_LOOP(x, y, z) for (size_t x = (y); x < (z); ++x)

#define MAX(a, b) (a) > (b) ? (a) : (b)

