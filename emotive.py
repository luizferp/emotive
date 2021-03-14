# Shared imports
import cv2 as cv
import mimetypes
from os import path
from face_attributes import AttrObj, AttrDict, Emotion, Age, ATTRIBUTES, ATTR_EMOTION, ATTR_AGE, ATTR_SMILE

# Imports for the Microsoft Emotion libs and dependencies
import requests
import json

TYPE_VIDEO = "video"
TYPE_IMAGE = "image"

DEBUG = False

class EmotiveError(Exception):
    """
    Handles all Emotive exceptions
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class FaceError(Exception):
    """
    Handles all Emotive exceptions
    """
    def __init__(self, code, value):
        self.code = code
        self.value = value

    def __str__(self):
        return "%s: %s"%(self.code, self.value)


class Face(object):
    """
    Defines face attributes, dimension and position
    Face:
    {
        id: <face-id>,
        top: <top>,
        left: <left>,
        width: <width>,
        height: <height>,
        faceAttributes: {
            emotion: {
                sadness: <sadness>,
                neutral: <neutral>,
                contempt: <contempt>,
                disgust: <disgust>,
                anger: <anger>,
                fear: <fear>,
                happiness: <happiness>,
            }
        }
    }

    """
    def __init__(self, face_attributes=None):
        self.top = 0
        self.left = 0
        self.width = 0
        self.height = 0
        self.__attributes = {}

        if face_attributes:
            self.add_attributes(face_attributes)
    
    def add_attributes(self, face_params):
        self.id = face_params.get('faceId')

        face_rect = face_params.get('faceRectangle')

        self.top = face_rect.get('top')
        self.left = face_rect.get('left')
        self.width = face_rect.get('width')
        self.height = face_rect.get('height')

        attributes = face_params.get('faceAttributes')
        
        for attr, attr_params in attributes.items():
            attr_class = ATTRIBUTES.get(attr)
            attr_instance = attr_class(attr_params)
            self.__attributes.update({attr: attr_instance})

    def __getattr__(self, arg):
        if arg in self.__attributes:
            attribute = self.__attributes.get(arg)
            if isinstance(attribute, dict):
                return attribute
            return attribute.value
        if arg in self.__dict__:
            return super(Face, self).__getattr__(arg)
        raise AttributeError("'%s' object has no attribute '%s'"%(Face.__name__, arg))


class Emotive(object):
    """
    Facade class for emotive APIs

    Emotive:
    {
        fps: <fps>,
        frames: [
            {
                frame_pos: <sec>,
                faces: [ <Face>, ]
            },
        ]
    }
    """
    MSFaceAPI = "Microsoft Face API"
    GVisionAPI = "Google Cloud Vision API"

    def __init__(self, file_name, backend_api, subscription_key=None):
        """
        backend_api is the API that will be used to detect emotions.
        Currently Microsoft Face API and Google Cloud Vision API are supported
        >>> emotive = Emotive(backend_api=Emotive.MSFaceAPI)
        >>> emotive = Emotive(backend_api=Emotive.GVisionAPI)
        """
        if (backend_api == Emotive.MSFaceAPI):
            if subscription_key:
                self.backend = MicrosoftEmotiveAPI(subscription_key)
            else:
                raise EmotiveError("MSFaceAPI requires a subscription key")
        elif (backend_api == Emotive.GVisionAPI):
            raise NotImplementedError("Google Cloud Vision API not yet implemented")
        else:
            raise EmotiveError("Unsupported backend API: " + backend_api)

        self.file_name = file_name
        mime, _ = mimetypes.guess_type(self.file_name)

        if (mime.startswith(TYPE_VIDEO)):
            self.__init_video__()
            self.type = TYPE_VIDEO
        else:
            self.type = TYPE_IMAGE
            self.__init_image__()

    def __init_video__(self):
        """
        Intended for internal use only. Do not call this method directly.
        """
        self.frames = []
        video = cv.VideoCapture(self.file_name)
        frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv.CAP_PROP_FPS)

        # video length in seconds
        video_length = int(round(frame_count / fps))

        frames_pos = [i for i in range(video_length)]
        for pos in frames_pos:
            self.frames.append(AttrDict({'frame_pos': pos, 'faces': []}))

        self.fps = fps

    def __init_image__(self):
        """
        Intended for internal use only. Do not call this method directly.
        """
        self.fps = 1
        self.frames = []
        self.frames.append(AttrDict({'frame_pos': 0, 'faces':[]}))

    def __detect_face_attributes(self, attrs=[ATTR_EMOTION]):
        if (self.type == TYPE_IMAGE):
            try:
                face_attrs = self.backend.detect_face_attributes_by_filename(self.file_name, attrs)
                for attr in face_attrs:
                    self.frames[0].faces.append(Face(attr))
            except FaceError as e:
                print("[ERROR] %s: %s"%(e.code, e.value))
        elif (self.type == TYPE_VIDEO):
            frames = self.frames
            video = cv.VideoCapture(self.file_name)
            for frame in frames:
                video_pos = frame.frame_pos * 1000
                video.set(cv.CAP_PROP_POS_MSEC, video_pos)
                success, image = video.read()
                conv = cv.imencode('.jpg', image)[1].tostring()
                try:
                    face_attrs = self.backend.detect_face_attributes(conv, attrs)
                    for attr in face_attrs:
                        frame.faces.append(Face(attr))
                except FaceError as e:
                    print("[ERROR] [%d] %s: %s"%(frame.frame_pos, e.code, e.value))

    def detect_emotion(self):
        self.__detect_face_attributes(attrs=[ATTR_EMOTION])
        
    def detect_age(self):
        self.__detect_face_attributes(attrs=[ATTR_AGE])
        
    def detect_smile(self):
        self.__detect_face_attributes(attrs=[ATTR_SMILE])

    def __save_to_video__(self, out_fname):

        # input video
        video = cv.VideoCapture(self.file_name)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        fps = video.get(cv.CAP_PROP_FPS)
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)

        # output video
        vout = cv.VideoWriter(out_fname, int(fourcc), fps, (width, height))

        frames = self.frames
        thickness = 2
        cur_frame_pos = 0
        for frame in frames:
            frame_pos = frame.frame_pos * fps # frame position in seconds * frames per second

            faces = frame.faces

            start_frame_pos = cur_frame_pos
            # go over all faces in the current frame and 
            # draw the rect and emotion on all faces up to the current frame
            for face in faces:
                top = face.top
                left = face.left
                width = face.width
                height = face.height
                emotion = face.emotion

                while cur_frame_pos <= frame_pos:
                    # read current frame
                    video.set(cv.CAP_PROP_POS_FRAMES, cur_frame_pos)
                    success, image = video.read()

                    # For the sake of simplicity, get the highest scored emotion
                    sorted_emotion = sorted(emotion.items(), key=lambda emo: emo[1], reverse=True)
                    text = "%s: %.2f"%(sorted_emotion[0][0], sorted_emotion[0][1])
                    (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    cv.rectangle(image, (left, top), (left+width, top+height),
                                 (0, 255, 0), thickness)

                    cv.putText(image, text, (left+th, top+height-th),
                                cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                                1, cv.LINE_AA)

                    vout.write(image)

                    cur_frame_pos = cur_frame_pos + 1

        vout.release()

    def __save_to_image__(self, out_fname):
        image = cv.imread(self.file_name)
        faces = self.frames[0].faces
        for face in faces:
            top = face.top
            left = face.left
            width = face.width
            height = face.height
            emotion = face.emotion

            # For the sake of simplicity, get the highest scored emotion
            sorted_emotion = sorted(emotion.items(), key=lambda emo: emo[1], reverse=True)
            text = "%s: %.2f"%(sorted_emotion[0][0], sorted_emotion[0][1])
            (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # draw the rectangle around the face
            cv.rectangle(image, (left, top), (left+width, top+height),
                            (0, 255, 0), 2)

            # draw the emotion text
            cv.putText(image, text, (left+th, top+height-th),
                    cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                    1, cv.LINE_AA)

        # write to the output file
        cv.imwrite(path.join(path.dirname(__file__), out_fname), image)

    def save(self, file_name=None):
        """
        """
        if file_name:
            out_fname = file_name
        else:
            out_fname = "out_"+self.file_name

        if (self.type == TYPE_IMAGE):
            self.__save_to_image__(out_fname)
        elif (self.type == TYPE_VIDEO):
            self.__save_to_video__(out_fname)

class IAttributeDetection(object):
    """
    Interface for the Emotion detection APIs
    """
    def __init__(self):
        pass

    def detect_face_attributes_by_filename(self):
        pass

    def __do_request(self):
        pass


class MicrosoftEmotiveAPI(IAttributeDetection):
    """
    It uses the Microsoft Emotive API for face detection
    """
    def __init__(self, subscription_key):
        self.key = subscription_key
        self.endpoint = "https://westcentralus.api.cognitive.microsoft.com"
        self.headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': self.key,
        }

        # Request parameters
        self.params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
        }

    def __do_request(self, urlparams, body):
        """
        Encapsulates network tasks
        """

        # Execute the REST API call and get the response.
        try:
            url = "/".join([self.endpoint, "face", "v1.0", "detect"])
            res = requests.post(url, params=urlparams, data=body, headers=self.headers)
            return res.text
        except Exception as e:
            return None  # TODO handle network exception correctly

    def detect_face_attributes(self, body, attrs=['emotion']):
        params = self.params.copy()
        params.update({'returnFaceAttributes': ','.join(attrs)})
        data = self.__do_request(params, body)

        parsed = json.loads(data)

        # 'data' contains the JSON body.
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            error = parsed.get('error')
            raise FaceError(error.get('code'), error.get('message'))

    def detect_face_attributes_by_filename(self, fname, attrs=['emotion']):
        with open(path.join(path.dirname(__file__), fname), "rb") as f:
            body = f.read()
            f.close()

        params = self.params.copy()
        params.update({'returnFaceAttributes': ','.join(attrs)})
        data = self.__do_request(params, body)

        # 'data' contains the JSON data.
        parsed = json.loads(data)
        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            error = parsed.get('error')
            raise FaceError(error.get('code'), error.get('message'))


class GoogleCloudVisionAPI(IAttributeDetection):

    def __init__(self):
        pass


if __name__ == "__main__":
    img_file = "happy1.jpg"
    em = Emotive(img_file, Emotive.MSFaceAPI, "")
    em.detect_emotion()
    em.save()

    video_file = "videoplayback.webm"
    em = Emotive(video_file, Emotive.MSFaceAPI, "")
    em.detect_emotion()
    em.save("out_playback.avi")
