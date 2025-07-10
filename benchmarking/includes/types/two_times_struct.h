#ifndef TIME_STRUCT
#define TIME_STRUCT

struct two_times_struct
{
    std::chrono::milliseconds cuda_time;
    std::chrono::milliseconds thrust_time;
};

struct three_times_struct
{
    std::chrono::milliseconds cuda_time;
    std::chrono::milliseconds thrust_time;
    std::chrono::milliseconds new_time;
};

struct four_times_struct
{
    std::chrono::milliseconds cuda_time;
    std::chrono::milliseconds thrust_time;
    std::chrono::milliseconds new_time;
    std::chrono::milliseconds new_fast_time;
};
#endif