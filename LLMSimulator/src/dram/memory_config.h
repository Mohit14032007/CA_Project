#pragma once

namespace llm_system {
class MemoryConfig {
 public:
 MemoryConfig& operator=(const MemoryConfig& rhs) = default;

  enum class MemoryType { HBM, DDR5 };

  // 16GB per cube, total 80GB HBM configuration & 1 cube, 16Gb density, 2 channel, LPDDR5
  MemoryConfig(int num_cube = 5, int num_logic_cube = 5, int num_channel = 32, int num_rank = 2,
               int num_bankgroup = 4, int num_bank = 4, int num_row = 16384,
               int num_col = 32, int granul = 32, MemoryType mem_type = MemoryType::HBM)
        : num_cube(num_cube),
          num_logic_cube(num_logic_cube),
          num_channel(num_channel),
          num_rank(num_rank),
          num_bankgroup(num_bankgroup),
          num_bank(num_bank),
          num_row(num_row),
          num_col(num_col),
          granul(granul),
          mem_type(mem_type){};

    int num_cube;
    int num_logic_cube;
    
    int num_channel;
    int num_rank;
    int num_bankgroup;
    int num_bank;
    int num_row;
    int num_col;
    int granul;
    MemoryType mem_type;
  };

  static MemoryConfig hbm3_80GB = MemoryConfig(
    5,      // num_cube 
    5,      // num_logic_cube
    32,     // num_channel
    2,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  static MemoryConfig hbm3e_192GB = MemoryConfig(
    8,      // num_cube 
    8,      // num_logic_cube
    32,     // num_channel
    3,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    16384,  // num_row
    32,     // num_col
    32     // granul
  );

  static MemoryConfig ddr5_32gb_x16_2ch = MemoryConfig(
    1,      // num_cube 
    1,      // num_logic_cube
    2,      // num_channel
    2,      // num_rank
    4,      // num_bankgroup
    4,      // num_bank
    10485760, // num_row (2^17) -> increased to simulate enough capacity for MoE
    64,     // num_col (2^10 / 16)
    64,     // granul (bytes per request)
    MemoryConfig::MemoryType::DDR5 // mem_type
  );

}  // namespace llm_system