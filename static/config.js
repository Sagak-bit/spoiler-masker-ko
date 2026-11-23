window.SPOILER_CONFIG = {
 "version": "23c4eb7d2f",
 "cache": "spoiler-models-23c4eb",
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
