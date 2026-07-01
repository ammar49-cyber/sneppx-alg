#ifndef _COVERAGE_H
#define _COVERAGE_H

// Standalone coverage instrumentation header.
// Usage: include in source files, then link with --coverage.

// Coverage counters — compile with -DCOVERAGE_ENABLED
#ifdef COVERAGE_ENABLED
    #define COV_HIT()   do { extern long long __cov_counter_##__LINE__; __cov_counter_##__LINE__++; } while(0)
    #define COV_BRANCH(c) do { extern long long __cov_branch_taken_##__LINE__; extern long long __cov_branch_not_##__LINE__; if(c) __cov_branch_taken_##__LINE__++; else __cov_branch_not_##__LINE__++; } while(0)
#else
    #define COV_HIT()
    #define COV_BRANCH(c) ((void)(c))
#endif

#endif
