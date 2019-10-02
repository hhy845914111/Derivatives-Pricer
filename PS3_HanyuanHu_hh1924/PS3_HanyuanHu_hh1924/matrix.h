#pragma once

using Matrix = double*;

inline Matrix cpu_matrix_malloc(size_t m, size_t n)
{
	return new double[m * n];
}

inline void cpu_matrix_free(Matrix& mat)
{
	delete[] mat;
	mat = nullptr;
}
