#include <iostream>
#include <iomanip>
#include <vector>
#include "dram/mmap_controller.h"
#include "dram/memory_config.h"

using namespace llm_system;

void print_mapping(MMapController::Ptr controller, addr address, const std::string& desc) {
    auto vec = controller->getAddrVec(address, 0);
    // Print the requested format
    std::cout << "Test: " << desc << "\n";
    std::cout << "  Logical byte address: " << address << "\n";
    std::cout << "  Cache-line address: " << (address / 64) << "\n";
    std::cout << "  Channel: " << vec[0] << "\n";
    std::cout << "  Rank: " << vec[1] << "\n";
    std::cout << "  Bank group: " << vec[2] << "\n";
    std::cout << "  Bank: " << vec[3] << "\n";
    std::cout << "  Row: " << vec[4] << "\n";
    std::cout << "  Column: " << vec[5] << "\n";
    bool valid = (vec[4] < 10485760); // It's valid in the *instantiated* Ramulator hierarchy
    std::cout << "  Valid in instantiated Ramulator: " << (valid ? "Yes" : "No") << "\n\n";
}

int main() {
    MemoryConfig config = ddr5_32gb_x16_2ch;
    auto mmap = MMapController::Create(config);
    
    long long GRANULE = 64;
    long long CUBE = 1;
    long long CH = 2;
    long long RA = 2;
    long long BG = 4;
    long long BA = 4;
    long long COL = 64;
    
    long long ROW_SIZE = CH * RA * BG * BA * COL * GRANULE; // 262144

    std::cout << "--- Address Translation Boundaries ---\n";
    print_mapping(mmap, 0, "1. Logical address zero");
    print_mapping(mmap, 64, "2. The next 64-byte cache line");
    print_mapping(mmap, 64 * CUBE, "3. A channel transition (assuming num_cube=1)");
    print_mapping(mmap, 64 * CUBE * CH, "4. A rank transition");
    print_mapping(mmap, 64 * CUBE * CH * RA, "5. A bank-group transition");
    print_mapping(mmap, 64 * CUBE * CH * RA * BG, "6. A bank transition");
    print_mapping(mmap, ROW_SIZE - 64, "7. The end of a row");
    print_mapping(mmap, 131072LL * ROW_SIZE, "8. The first address after the original 131,072-row capacity boundary");
    print_mapping(mmap, 5000000LL * ROW_SIZE, "9. An address several million rows into the expanded range");
    
    // 14592 experts, 88080384 bytes each
    long long total_experts = 14592;
    long long expert_size = 88080384;
    // Assuming experts start exactly at 0 (or some offset, we just pick 0 for illustration)
    long long late_expert_start = 14000LL * expert_size;
    long long late_expert_end = late_expert_start + expert_size - 1;
    print_mapping(mmap, late_expert_start, "10a. The start address of a late DeepSeek-V3 expert (Expert 14000)");
    print_mapping(mmap, late_expert_end, "10b. The end address of a late DeepSeek-V3 expert");
    
    long long highest_addr = total_experts * expert_size - 1;
    print_mapping(mmap, highest_addr, "11. The highest logical address occupied by routed expert weights");
    
    return 0;
}
