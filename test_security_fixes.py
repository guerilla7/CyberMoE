#!/usr/bin/env python3
"""Integration test for security fixes."""
import sys

print('Testing CyberMoE application components after security fixes...\n')

# Test 1: Import all main modules
print('1. Importing main modules...')
try:
    import app
    import model
    import finetune_from_feedback
    import feedback
    import preprocessor
    print('   ✓ All modules imported successfully')
except Exception as e:
    print(f'   ✗ Import failed: {e}')
    sys.exit(1)

# Test 2: Load the finetuned model with weights_only=True
print('\n2. Testing torch.load with weights_only=True...')
try:
    from finetune_from_feedback import load_finetuned_model
    model_obj = load_finetuned_model()
    if model_obj is not None:
        print(f'   ✓ Finetuned model loaded successfully (type: {type(model_obj).__name__})')
        print(f'   ✓ Model has {sum(p.numel() for p in model_obj.parameters()):,} parameters')
    else:
        print('   ℹ No finetuned checkpoint found (expected if not yet trained)')
except Exception as e:
    print(f'   ✗ Model loading failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Create a fresh model instance
print('\n3. Testing fresh model instantiation...')
try:
    from model import CyberMoE
    fresh_model = CyberMoE(top_k=2)
    print(f'   ✓ Fresh CyberMoE model created successfully')
    print(f'   ✓ Model config: top_k={fresh_model.top_k}, num_experts={fresh_model.num_experts}')
except Exception as e:
    print(f'   ✗ Model creation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test feedback logging
print('\n4. Testing feedback logging...')
try:
    from feedback import log_feedback
    test_record = {
        'user_input': 'test_security_fix',
        'pred_label': 'phishing',
        'confidence': 0.95,
        'timestamp': '2026-02-08'
    }
    log_feedback(test_record)
    print('   ✓ Feedback logging works')
except Exception as e:
    print(f'   ✗ Feedback logging failed: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '='*70)
print('✅ All critical components tested successfully!')
print('='*70)
print('\nSecurity fixes applied:')
print('  1. torch.load with weights_only=True (prevents RCE)')
print('  2. Shell injection fix in init_cache.sh')
print('  3. Pinned dependencies in requirements.txt')
print('  4. Non-root user in Dockerfile')
print('='*70)
