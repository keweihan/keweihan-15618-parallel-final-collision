#pragma once

#ifdef _WIN32
  #ifdef SIMPLEECS_EXPORTS
    // Exporting symbols when building the DLL
    #define SIMPLEECS_API __declspec(dllexport)
  #else
    // Importing symbols when using the DLL
    #define SIMPLEECS_API __declspec(dllimport)
  #endif
#else
  // Non-Windows platforms
  #define SIMPLEECS_API
#endif