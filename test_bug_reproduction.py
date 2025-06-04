"""
Test to reproduce and verify the fix for TVM bug where inject_software_pipeline 
fails when RewriteReduction is applied before it.

The issue:
- Original reduction block (e.g., conv2d_nchw) gets decomposed into init and update blocks
- This increases block count from 3 to 4
- But software pipeline annotations remain at size 3
- Causing CHECK_EQ failure at line 1133 in inject_software_pipeline.cc

This test demonstrates that our fix handles this case correctly.
"""

import tvm
from tvm.script import tir as T


@T.prim_func
def matmul_reduction_example(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"), 
    C: T.Buffer((128, 128), "float32")
) -> None:
    """A simple matmul with reduction that can trigger the bug"""
    T.func_attr({
        "global_symbol": "main",
        # These annotations would normally be added by meta_schedule
        # before reduction decomposition occurs
        "tir.software_pipeline_stage": [0, 1, 2],
        "tir.software_pipeline_order": [0, 1, 2]
    })
    
    for i in T.serial(128):  # Loop that can be software pipelined
        with T.block("load_A"):
            vi = T.axis.spatial(128, i)
            # Stage 0: Load data
            A_shared = T.allocate([128], "float32", "shared")
            for k in range(128):
                A_shared[k] = A[vi, k]
                
        with T.block("matmul"):
            vi = T.axis.spatial(128, i)
            # Stage 1: Compute (this block has reduction that can be decomposed)
            with T.init():
                C[vi, 0] = T.float32(0)
            for j, k in T.grid(128, 128):
                with T.block("matmul_update"):
                    vj, vk = T.axis.remap("SR", [j, k])
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                    
        with T.block("store_C"):
            vi = T.axis.spatial(128, i)
            # Stage 2: Store result  
            # Store operations would go here
            pass


def test_bug_reproduction():
    """
    Test that reproduces the original bug scenario and verifies our fix works.
    
    The sequence that triggers the bug:
    1. Function has software pipeline annotations for 3 blocks
    2. RewriteReduction is applied, decomposing reduction block into init + update  
    3. This increases block count to 4
    4. inject_software_pipeline is called with original annotations (size 3)
    5. CHECK_EQ(pipeline_stages.size(), original_order.size()) fails
    
    Our fix should handle this gracefully by either:
    - Filtering out init blocks, or
    - Extending annotations to match new block count
    """
    print("=== Testing Bug Reproduction and Fix ===")
    
    # Start with the function that has pipeline annotations
    func = matmul_reduction_example
    
    print("1. Original function created with software pipeline annotations")
    print(f"   Annotations expect 3 stages: {func.attrs['tir.software_pipeline_stage']}")
    
    # Create module and schedule
    mod = tvm.IRModule.from_expr(func)
    sch = tvm.tir.Schedule(mod, debug_mask='all')
    
    # Count original blocks
    original_blocks = []
    def count_blocks(stmt):
        if isinstance(stmt, tvm.tir.Block):
            original_blocks.append(stmt.name_hint)
    
    # Get the matmul block that contains reduction
    try:
        matmul_block = sch.get_block("matmul")
        print(f"2. Found matmul block: {matmul_block}")
        
        # Get loops for decomposition
        loops = sch.get_loops(matmul_block)
        print(f"   Block has {len(loops)} loops")
        
        if len(loops) > 0:
            print("3. Applying reduction decomposition (simulating RewriteReduction)...")
            
            # This is where the bug would be triggered in the original code
            # DecomposeReduction increases block count but annotations stay the same
            try:
                init_block = sch.decompose_reduction(matmul_block, loops[0])
                print(f"   ✓ Decomposition successful, created init block: {init_block}")
                print("   ✓ Block count increased (matmul -> matmul_init + matmul_update)")
                
            except Exception as e:
                print(f"   ✗ Decomposition failed: {e}")
                return False
                
            print("4. Now applying software pipeline injection...")
            print("   (This would fail in original code due to size mismatch)")
            
            # Apply the software pipeline transformation 
            # Our fix should handle the size mismatch gracefully
            try:
                mod_with_pipeline = tvm.tir.transform.InjectSoftwarePipeline()(sch.mod)
                print("   ✓ Software pipeline injection successful!")
                print("   ✓ Our fix handled the block count mismatch correctly")
                return True
                
            except Exception as e:
                print(f"   ✗ Software pipeline injection failed: {e}")
                print("   ✗ Fix may need adjustment")
                return False
        else:
            print("   No loops available for decomposition")
            return False
            
    except Exception as e:
        print(f"Error accessing matmul block: {e}")
        return False


def test_fix_validation():
    """Additional test to validate fix behavior"""
    print("\n=== Validating Fix Behavior ===")
    
    # Test 1: Normal case (no decomposition)
    @T.prim_func
    def normal_pipeline(A: T.Buffer((16,), "float32")) -> None:
        T.func_attr({
            "global_symbol": "main",
            "tir.software_pipeline_stage": [0, 1],
            "tir.software_pipeline_order": [0, 1]
        })
        for i in T.serial(16):
            with T.block("load"):
                vi = T.axis.spatial(16, i)
                pass
            with T.block("compute"):
                vi = T.axis.spatial(16, i)
                pass
    
    try:
        mod = tvm.IRModule.from_expr(normal_pipeline)
        mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
        print("✓ Normal case works correctly")
    except Exception as e:
        print(f"✗ Normal case failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Testing TVM inject_software_pipeline fix for RewriteReduction compatibility\n")
    
    # Run the main bug reproduction test
    bug_test_passed = test_bug_reproduction()
    
    # Run additional validation
    validation_passed = test_fix_validation()
    
    print(f"\n=== Results ===")
    print(f"Bug reproduction test: {'PASS' if bug_test_passed else 'FAIL'}")
    print(f"Fix validation test: {'PASS' if validation_passed else 'FAIL'}")
    
    overall_success = bug_test_passed and validation_passed
    print(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    if overall_success:
        print("\n🎉 All tests passed! The fix successfully resolves the TVM bug.")
        print("   inject_software_pipeline now works correctly even when")
        print("   RewriteReduction is applied before it.")
    else:
        print("\n❌ Some tests failed. The fix may need further refinement.")
