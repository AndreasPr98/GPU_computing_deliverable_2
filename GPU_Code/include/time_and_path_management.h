#ifndef TIME_AND_PATH_MANAGEMENT_H
#define TIME_AND_PATH_MANAGEMENT_H

#include <sys/time.h>
#include <stdio.h>
#include <filesystem>
#include <vector>
#include <string>
#include "utils.h"
#include "parser.h"

#define TIMER_DEF(n)	 struct timespec time_1_##n={0,0}, time_2_##n={0,0}
#define TIMER_START(n)	 clock_gettime(CLOCK_MONOTONIC, &time_1_##n)
#define TIMER_STOP(n)	 clock_gettime(CLOCK_MONOTONIC, &time_2_##n)
#define TIMER_ELAPSED(n) ((time_2_##n.tv_sec-time_1_##n.tv_sec)*1.e9+(time_2_##n.tv_nsec-time_1_##n.tv_nsec))
#define TIMER_PRINT(n) \
    do { \
        printf("Timer elapsed: %.9f\n", TIMER_ELAPSED(n)/1e9);\
        fflush(stdout);\
    } while (0);

bool handle_path_management(int argc, char* argv[], 
                            const std::filesystem::path& base_path, 
                            const std::filesystem::path& target_dir, 
                            std::filesystem::directory_entry& out_entry);

#endif
