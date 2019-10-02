// dll_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <windows.h>
#include <iostream>

using namespace std;

typedef double(*p_func)(double y0, double y1, double tao);
typedef double(*a_func)(double x, double y);

int main(int argc, char *argv[])
{
	HMODULE hDll = LoadLibrary(L"C:\\Users\\hoore\\Documents\\workspace\\GPU\\rate_dll2\\x64\\Release\\rate_dll2.dll");
	if (hDll != NULL)
	{
		p_func func = (p_func)GetProcAddress(hDll, "P");
		if (func != NULL)
		{
			cout << func(0.1, 0.1, 2) << endl;
		}
		FreeLibrary(hDll);
	}
}