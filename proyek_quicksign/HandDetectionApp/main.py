from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
from handDetection import HandDetection

class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.handDetection = HandDetection(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            handLandMarks = self.handDetection.findHandLandMarks(image=frame, draw=True)
            
            # Convert frame to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = image_texture

    def on_stop(self):
        self.capture.release()

class MainApp(App):
    def build(self):
        layout = BoxLayout()
        self.camera = KivyCamera()
        layout.add_widget(self.camera)
        return layout

    def on_stop(self):
        self.camera.on_stop()

if __name__ == '__main__':
    MainApp().run()
