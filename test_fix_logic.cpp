#include <iostream>
#include <vector>
#include <string>
#include <cassert>

// Simplified test to validate the logic of our fix for handling 
// reduction decomposition in software pipeline injection

struct Block {
    std::string name_hint;
    explicit Block(const std::string& name) : name_hint(name) {}
};

// Simulate the fix logic
std::pair<std::vector<Block>, std::vector<int>> handle_reduction_decomposition(
    const std::vector<Block>& original_blocks,
    const std::vector<int>& pipeline_stages) {
    
    std::vector<Block> processed_blocks = original_blocks;
    std::vector<int> processed_stages = pipeline_stages;
    
    if (pipeline_stages.size() != original_blocks.size()) {
        // Detect if this mismatch is due to reduction decomposition
        std::vector<bool> is_init_block(original_blocks.size(), false);
        int init_block_count = 0;
        
        for (size_t i = 0; i < original_blocks.size(); ++i) {
            const std::string& block_name = original_blocks[i].name_hint;
            if (block_name.length() > 5 && 
                block_name.substr(block_name.length() - 5) == "_init") {
                is_init_block[i] = true;
                init_block_count++;
            }
        }
        
        // If the number of non-init blocks matches the annotation size,
        // filter out init blocks and use original annotations
        if (original_blocks.size() - init_block_count == pipeline_stages.size()) {
            std::vector<Block> filtered_blocks;
            for (size_t i = 0; i < original_blocks.size(); ++i) {
                if (!is_init_block[i]) {
                    filtered_blocks.push_back(original_blocks[i]);
                }
            }
            processed_blocks = filtered_blocks;
            // Keep original annotations unchanged
        } else {
            // If filtering doesn't solve the mismatch, extend annotations
            std::vector<int> extended_stages;
            
            for (size_t i = 0; i < original_blocks.size(); ++i) {
                if (i < pipeline_stages.size()) {
                    extended_stages.push_back(pipeline_stages[i]);
                } else {
                    // For extra blocks, use the last annotation value
                    extended_stages.push_back(pipeline_stages.back());
                }
            }
            
            processed_stages = extended_stages;
        }
    }
    
    return {processed_blocks, processed_stages};
}

void test_no_decomposition() {
    std::cout << "Test 1: No decomposition (sizes match)" << std::endl;
    
    std::vector<Block> blocks = {Block("block1"), Block("block2"), Block("block3")};
    std::vector<int> stages = {0, 1, 2};
    
    auto [result_blocks, result_stages] = handle_reduction_decomposition(blocks, stages);
    
    assert(result_blocks.size() == 3);
    assert(result_stages.size() == 3);
    assert(result_blocks.size() == result_stages.size());
    
    std::cout << "✓ Passed" << std::endl;
}

void test_with_init_blocks() {
    std::cout << "Test 2: Reduction decomposition with init blocks" << std::endl;
    
    // Simulate what happens after reduction decomposition:
    // Original: [conv2d_nchw] with 1 annotation
    // After decomposition: [conv2d_nchw_init, conv2d_nchw_update] with 1 annotation
    std::vector<Block> blocks = {Block("conv2d_nchw_init"), Block("conv2d_nchw_update")};
    std::vector<int> stages = {0}; // Only 1 original annotation
    
    auto [result_blocks, result_stages] = handle_reduction_decomposition(blocks, stages);
    
    // Should filter out init blocks, leaving only update block
    assert(result_blocks.size() == 1);
    assert(result_stages.size() == 1);
    assert(result_blocks[0].name_hint == "conv2d_nchw_update");
    assert(result_blocks.size() == result_stages.size());
    
    std::cout << "✓ Passed" << std::endl;
}

void test_extension_fallback() {
    std::cout << "Test 3: Extension fallback when filtering doesn't work" << std::endl;
    
    // Case where we have more blocks but they're not all init blocks
    std::vector<Block> blocks = {Block("block1"), Block("block2"), Block("block3"), Block("block4")};
    std::vector<int> stages = {0, 1, 2}; // 3 annotations, 4 blocks
    
    auto [result_blocks, result_stages] = handle_reduction_decomposition(blocks, stages);
    
    // Should extend annotations to match block count  
    assert(result_blocks.size() == 4);
    assert(result_stages.size() == 4);
    assert(result_stages[3] == 2); // Last block gets last annotation value
    assert(result_blocks.size() == result_stages.size());
    
    std::cout << "✓ Passed" << std::endl;
}

void test_complex_decomposition() {
    std::cout << "Test 4: Complex case with multiple reductions" << std::endl;
    
    // Multiple blocks with some decomposed
    std::vector<Block> blocks = {
        Block("load"), 
        Block("conv2d_nchw_init"), 
        Block("conv2d_nchw_update"),
        Block("matmul_init"),
        Block("matmul_update"), 
        Block("store")
    };
    std::vector<int> stages = {0, 1, 2, 3}; // 4 original annotations, 6 blocks after decomposition
    
    auto [result_blocks, result_stages] = handle_reduction_decomposition(blocks, stages);
    
    // Should filter out init blocks
    assert(result_blocks.size() == 4);
    assert(result_stages.size() == 4);
    assert(result_blocks.size() == result_stages.size());
    
    // Check that init blocks were filtered out
    for (const auto& block : result_blocks) {
        if (block.name_hint.length() > 5) {
            assert(block.name_hint.substr(block.name_hint.length() - 5) != "_init");
        }
    }
    
    std::cout << "✓ Passed" << std::endl;
}

int main() {
    std::cout << "=== Testing Reduction Decomposition Fix Logic ===" << std::endl;
    std::cout << std::endl;
    
    try {
        test_no_decomposition();
        test_with_init_blocks();
        test_extension_fallback();
        test_complex_decomposition();
        
        std::cout << std::endl;
        std::cout << "✓ All tests passed! The fix logic is working correctly." << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Test failed: " << e.what() << std::endl;
        return 1;
    }
}
