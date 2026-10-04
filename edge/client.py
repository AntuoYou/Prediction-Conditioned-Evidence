"""边缘端 -> 云端 HTTP 客户端"""

import time, requests
from typing import List

class CloudClient:
    def __init__(self, url="http://127.0.0.1:8000", timeout=10.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def health(self) -> bool:
        try:
            return self.s.get(f"{self.url}/health", timeout=3).status_code == 200
        except: return False

    def detect(self, image_bytes: bytes, roi_index: int = 0,
                conf_threshold: float = 0.3) -> dict:
        try:
            r = self.s.post(f"{self.url}/detect",
                files={"file": ("roi.jpg", image_bytes, "image/jpeg")},
                data={"roi_index": str(roi_index),
                      "conf_threshold": str(conf_threshold)},
                timeout=self.timeout)
            r.raise_for_status()
            result = r.json()
            result["network_time_ms"] = round((time.time()-result.get("_t",time.time()))*1000, 2)
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "detections": []}

    def batch_detect(self, reqs: List[dict], conf_threshold=0.3) -> List[dict]:
        results = []
        for req in reqs:
            t0 = time.time()
            r = self.detect(req["image_bytes"], req["index"], conf_threshold)
            r["network_time_ms"] = round((time.time()-t0)*1000, 2)
            r["upload_size_kb"] = round(len(req["image_bytes"])/1024, 1)
            r["original_bbox"] = req["original_bbox"]
            r["crop_bbox"] = req.get("crop_bbox")
            r["edge_metadata"] = req["metadata"]
            r["low_conf_index"] = req.get("index")
            results.append(r)
        ok = sum(1 for r in results if r.get("success"))
        print(f"[Cloud] 完成: {ok}/{len(results)} 成功")
        return results
