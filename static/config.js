window.SPOILER_CONFIG = {
 "version": "8923560b67",
 "cache": "spoiler-models-892356",
 "vocab": "data/vocab.txt",
 "iif": "data/iif.json",
 "models": {
  "m5": {
   "model": "https://huggingface.co/dPckslim/koelectra-spoiler-ko/resolve/main/onnx/m5_full_s42.int8.onnx",
   "spec": "data/m5_full_s42.json",
   "mb": 124
  },
  "m1": {
   "model": "https://huggingface.co/dPckslim/koelectra-spoiler-ko/resolve/main/onnx/m1_ssc_s42.int8.onnx",
   "spec": "data/m1_ssc_s42.json",
   "mb": 113
  }
 }
};
