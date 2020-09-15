import glob
import os
import xml.etree.ElementTree as et
from datetime import timedelta
from typing import Any, Dict, List, Set, Optional

import ffmpeg
import numpy as np

from .video import *


class MetaData():
    def __init__(self, filename: str):
        super().__init__()
        self.__filename = os.path.abspath(filename)
        self.__root = et.parse(filename).getroot()

    @property
    def objects(self) -> List["Object"]:
        return [Object(self, tree) for tree in self.__root.findall("object")]

    @property
    def video_filename(self) -> str:
        return os.path.join(
            os.path.split(self.filename)[0], self.__root.findtext("filename"))

    @property
    def filename(self) -> str:
        return self.__filename

    @property
    def class_name(self) -> str:
        return self.__root.find("event").findtext("eventname")

    @property
    def starttime(self) -> str:
        return self.__root.find("event").findtext("starttime")

    @property
    def duration(self) -> str:
        return self.__root.find("event").findtext("duration")

    @property
    def other_info(self) -> Dict[str, Any]:
        """header와 size 정보 반환
        """
        els = self.__root.find("header").getchildren() + self.__root.find(
            "size").getchildren()
        return {el.tag: el.text for el in els}


class Object():
    def __init__(self, metadata: MetaData, tree: et):
        super().__init__()
        self.__metadata = metadata
        self.__tree = tree

    @property
    def name(self) -> str:
        return self.__tree.findtext("objectname")

    @property
    def position(self) -> Dict[str, Any]:
        return {
            "keyframe":
            self.__tree.find("position").findtext("keyframe"),
            "keypoint": (self.__tree.find("position/keypoint").findtext("x"),
                         self.__tree.find("position/keypoint").findtext("x"))
        }

    @property
    def actions(self) -> List["Action"]:
        tree_a = self.__tree.find("action")
        name = tree_a.findtext("actionname")
        return [
            Action(self, name, int(tree.findtext("start")),
                   int(tree.findtext("end")))
            for tree in tree_a.findall("frame")
        ]

    @property
    def metadata(self) -> MetaData:
        return self.__metadata


class Action():
    def __init__(self, actor: Object, name: str, start: int, end: int):
        super().__init__()
        self.__actor = actor
        self.__name = name
        self.__start = start
        self.__end = end
        self.__fps = float(self.__actor.metadata.other_info['fps'])

    @property
    def name(self) -> str:
        return self.__name

    @property
    def starttime(self) -> float:
        return self.__start / self.__fps

    @property
    def starttime_str(self) -> str:
        return self.__sec_to_str(self.starttime)

    @property
    def duration(self) -> float:
        return (self.__end - self.__start) / self.__fps

    @property
    def duration_str(self) -> str:
        return self.__sec_to_str(self.duration)

    @property
    def actor(self) -> Object:
        return self.__actor

    @staticmethod
    def __sec_to_str(sec: float) -> str:
        return str(timedelta(milliseconds=sec * 1000))


def get_all_metadata(root_path: str) -> List[MetaData]:
    xmls = glob.glob(os.path.join(root_path, "**/*.xml"), recursive=True)
    return [MetaData(xml) for xml in xmls]


def get_action_names_from_action_list(lst: List[Action]) -> Set[str]:
    return set([act.name for act in lst])


def get_action_names_from_metadata_list(lst: List[MetaData]) -> Set[str]:
    return get_action_names_from_action_list(
        [act for meta in lst for obj in meta.objects for act in obj.actions])


def create_clip_from_action(action: Action,
                            fps: float = 5,
                            explicit_duration: Optional[float] = None,
                            resolution: str = "768x480") -> np.ndarray:
    filename = action.actor.metadata.video_filename
    duration = action.duration
    if explicit_duration:
        duration = explicit_duration
    return create_clip(filename, action.starttime, duration, fps, resolution)
