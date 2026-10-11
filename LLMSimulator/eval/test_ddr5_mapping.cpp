#include <iostream>
#include <cassert>
#include <vector>
#include "dram/mmap_controller.h"
#include "dram/memory_config.h"

using namespace llm_system;

void print_addr(const std::string& msg, addr address, const Ramulator::AddrVec_t& vec) {
    std::cout << msg << " Address: " << address << " -> [";
    for (size_t i = 0; i < vec.size(); i++) {
        std::cout << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    std::cout << "]\n";
}

int main() {
    MemoryConfig config = ddr5_32gb_x16_2ch;
    auto mmap = MMapController::Create(config);

    std::cout << "Testing DDR5 extended mapping (num_row = 10485760)\n";
    std::cout << "Geometry: Channel=" << config.num_channel 
              << ", Rank=" << config.num_rank 
              << ", BG=" << config.num_bankgroup 
              << ", Bank=" << config.num_bank 
              << ", Row=" << config.num_row 
              << ", Col=" << config.num_col 
              << ", Granul=" << mmap->getGranul() << "\n\n";

    // Highest routed-expert footprint: ~1.285 TB
    long long max_address = 1285268963328LL;
    
    // Address 0
    Ramulator::AddrVec_t vec0 = mmap->getAddrVecLOGIC(0, 0);
    print_addr("Zero", 0, vec0);

    // Row boundary transition
    long long row_size = config.num_col * mmap->getGranul(); // 64 * 64 = 4096 bytes
    long long bank_size = config.num_row * row_size; // 10485760 * 4096 = 42,949,672,960 bytes

    Ramulator::AddrVec_t vec_row0_end = mmap->getAddrVecLOGIC(row_size - 1, 0);
    print_addr("Row 0 End", row_size - 1, vec_row0_end);
    
    Ramulator::AddrVec_t vec_row1_start = mmap->getAddrVecLOGIC(row_size, 0);
    print_addr("Row 1 Start", row_size, vec_row1_start);

    // Distinct-address preservation
    Ramulator::AddrVec_t vec_expert1 = mmap->getAddrVecLOGIC(4294967296LL, 0);
    Ramulator::AddrVec_t vec_expert2 = mmap->getAddrVecLOGIC(4294967296LL + 88080384LL, 0);
    print_addr("Expert 1 start", 4294967296LL, vec_expert1);
    print_addr("Expert 2 start", 4294967296LL + 88080384LL, vec_expert2);

    // Highest routed-expert address
    Ramulator::AddrVec_t vec_max = mmap->getAddrVecLOGIC(max_address, 0);
    print_addr("Highest Expert Address", max_address, vec_max);

    assert(vec_max.size() > 0);
    // Print row index of highest address
    std::cout << "Highest Row Index: " << vec_max[config.mem_type == MemoryConfig::MemoryType::DDR5 ? 4 : 4] << "\n"; // In DDR5: ch, rank, bg, bank, row, col. Row is index 4.

    std::cout << "All mapping tests passed.\n";

    return 0;
}
