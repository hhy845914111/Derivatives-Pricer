#pragma once

constexpr auto GPU_TO_FILE = "./GPU_exp.csv";
constexpr auto CPU_TO_FILE = "./CPU_exp.csv";

constexpr size_t TOTAL_SIM_NUM(1000000);
constexpr size_t CPU_THREAD_COUNT(8);
constexpr int SEEDS_SIZE(6);
constexpr size_t GPU_THREAD_COUNT(128);
enum OptionType { Call, Put };

constexpr size_t BOOTSTRAP_COUNT(10);

constexpr bool SAVE = true;