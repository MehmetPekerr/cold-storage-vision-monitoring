# engine/modules/base.py — BaseModule arayüzü

class BaseModule:
    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        raise NotImplementedError

    def get_data(self) -> dict:
        raise NotImplementedError

    def draw(self, frame):
        return frame      # varsayılan: frame'e dokunmaz

    def reset(self):
        pass              # isteğe bağlı

    def shutdown(self):
        pass              # isteğe bağlı
