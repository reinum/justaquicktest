import io
import clr
import json
clr.AddReference('OsuParsers')
import OsuParsers
from OsuParsers.Beatmaps import *
from OsuParsers.Decoders import *

def findHitObject(bMap):
    beatmap = BeatmapDecoder.Decode(bMap)

    output = []
    for hitObject in list(beatmap.get_HitObjects()):
        hitobject = {}
        hittype = str(hitObject.GetType()).split(".")[-1]

        if hittype == "Slider":
            hitobject["Type"] = "slider"
            hitobject["X"] = hitObject.get_Position().X
            hitobject["Y"] = hitObject.get_Position().Y
            hitobject["Time"] = hitObject.get_StartTime()
            hitobject["HoldTime"] = (hitObject.get_EndTime() - hitObject.get_StartTime())
            hitobject["CurveType"] = str(hitObject.get_CurveType())
            hitobject["SliderPoints"] = [(point.X, point.Y) for point in hitObject.get_SliderPoints()]
            hitobject["RepeatCount"] = hitObject.get_Repeats()
            hitobject["PixelLength"] = hitObject.get_PixelLength()
            hitobject["EndTime"] = hitObject.get_EndTime()

        elif hittype == "HitCircle":
            hitobject["Type"] = "hitcircle"
            hitobject["X"] = hitObject.get_Position().X
            hitobject["Y"] = hitObject.get_Position().Y
            hitobject["Time"] = hitObject.get_StartTime()
            
        elif hittype == "Spinner":
            hitobject["Type"] = "spinner"
            hitobject["X"] = hitObject.get_Position().X
            hitobject["Y"] = hitObject.get_Position().Y
            hitobject["Time"] = hitObject.get_StartTime()
            hitobject["HoldTime"] = (hitObject.get_EndTime() - hitObject.get_StartTime())
            hitobject["EndTime"] = hitObject.get_EndTime()

        output.append(hitobject)

    return output