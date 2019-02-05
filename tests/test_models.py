from salad.models import DigitModel

def test_conditional_model():

    model = DigitModel(n_domains=2)

    all_params_0    = set(model.parameters(0, yield_shared=True, yield_conditional=True))
    all_params_1    = set(model.parameters(1, yield_shared=True, yield_conditional=True))

    shared        = set(model.parameters(0, yield_shared=True,  yield_conditional=False))
    conditional_0 = set(model.parameters(0, yield_shared=False, yield_conditional=True))
    conditional_1 = set(model.parameters(1, yield_shared=False, yield_conditional=True))

    print (len(conditional_0.union(conditional_1).union(shared)), len(all_params_0.union(all_params_1)))

    assert conditional_0.union(conditional_1).union(shared) == all_params_0.union(all_params_1)
    assert not (conditional_0.intersection(conditional_1))
    assert all_params_0.union(conditional_1) == all_params_1.union(conditional_0)
    assert all_params_0.intersection(all_params_1) == shared