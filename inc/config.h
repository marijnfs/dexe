#pragma once

#if defined(_MSC_VER)
#	if defined(DEXE_DLL)
#		define DEXE_API __declspec(dllexport)
#	elif defined(DEXE_STATIC)
#		define DEXE_API
#	else
#		define DEXE_API __declspec(dllimport)
#	endif
#else
#	if defined(DEXE_DLL)
#		define DEXE_API __attribute__((visibility("default")))
#	elif defined(DEXE_STATIC)
#		define DEXE_API
#	else
#		define DEXE_API __attribute__((visibility("default")))
#	endif
#endif