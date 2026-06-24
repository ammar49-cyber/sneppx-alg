# CMake generated Testfile for 
# Source directory: C:/Users/PC/ARIX/ARIX_Algo/tests
# Build directory: C:/Users/PC/ARIX/ARIX_Algo/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test_tensor "C:/Users/PC/ARIX/ARIX_Algo/build/tests/Debug/test_tensor.exe")
  set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;8;add_test;C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test_tensor "C:/Users/PC/ARIX/ARIX_Algo/build/tests/Release/test_tensor.exe")
  set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;8;add_test;C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test_tensor "C:/Users/PC/ARIX/ARIX_Algo/build/tests/MinSizeRel/test_tensor.exe")
  set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;8;add_test;C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test_tensor "C:/Users/PC/ARIX/ARIX_Algo/build/tests/RelWithDebInfo/test_tensor.exe")
  set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;8;add_test;C:/Users/PC/ARIX/ARIX_Algo/tests/CMakeLists.txt;0;")
else()
  add_test(test_tensor NOT_AVAILABLE)
endif()
