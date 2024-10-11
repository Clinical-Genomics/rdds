# Testing Model Saving/ Loading

## Keras Models

Make sure to test models in the following manner to really
test the ins-and-outs of model serialization and deserialization.

```python

tf.keras.models.Model.from_config(model.get_config)

tf.keras.models.clone_model(model)

model.save(work_dir)
/.../
tf.keras.saving.load_model(work_dir)
```
