from pathlib import Path
import joblib
p=Path('d:/house/models/trained/preprocessor.pkl')
print('Loading',p)
pre=joblib.load(p)
print('Type:',type(pre))

# Try get_feature_names_out
try:
    names = pre.get_feature_names_out()
    print('get_feature_names_out():')
    print(names)
except Exception as e:
    print('get_feature_names_out failed:',e)
    # Inspect transformers_
    if hasattr(pre,'transformers_'):
        for name, trans, cols in pre.transformers_:
            print('Transformer:',name,'cols=',cols)
            try:
                fn = trans.get_feature_names_out(cols if cols is not None else [])
                print('-> produced:',fn)
            except Exception as ee:
                print('-> get_feature_names_out failed for transformer:',ee)

if hasattr(pre,'steps'):
    print('Pipeline steps:', [s[0] for s in pre.steps])
