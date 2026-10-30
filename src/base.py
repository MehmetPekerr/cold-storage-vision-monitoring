
class BaseModule:
    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        raise NotImplementedError

    def get_data(self) -> dict:
        raise NotImplementedError

    def draw(self, frame):
        return frame

    def reset(self):
        pass

    def shutdown(self):
        pass
