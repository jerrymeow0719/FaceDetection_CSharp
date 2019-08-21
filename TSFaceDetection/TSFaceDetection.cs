using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;
using Emgu;
using Emgu.CV;
using FaceDetect_EmguCV;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Newtonsoft.Json;

namespace TSFaceDetection
{
    /// <summary>
    /// 
    /// </summary>
    public class TSFaceDetection
    {
        /// <summary>
        /// Face detection using Cascade classifier
        /// </summary>
        /// <param name="FullFileName"> Full input file name</param>
        /// <param name="JsonFileName"> Output JSon file name (saving faces information)</param>
        /// <param name="FaceFileName_Cascade"> Cascade config file (.xml)</param>
        /// <param name="DetectInterval"> Detect interval</param>
        public void FaceDetect(string FullFileName, string JsonFileName, string FaceFileName_Cascade, int DetectInterval)
        {
            if (File.Exists(JsonFileName))
                File.Delete(JsonFileName);
            string FileFormat = System.IO.Path.GetExtension(FullFileName);
            FileFormat = FileFormat.ToLower();
            OpenCVResult openCVResult = new OpenCVResult();
            TSDic.Clear();
            if (FileFormat == ".jpg")
            {
                using (Emgu.CV.Mat PictureMat = CvInvoke.Imread(FullFileName))
                {
                    openCVResult = FaceDetect_Cascade(PictureMat, FaceFileName_Cascade);
                    List<TSJSonClass> TSFaces = new List<TSJSonClass>();
                    for (int i = 0; i < openCVResult.faces.Count; i++)
                    {
                        TSFaces.Add(new TSJSonClass { face = openCVResult.faces[i], enable = true });
                        GC.Collect();
                    }
                    TSDic.Add(1.ToString("00000"), TSFaces); //Image always uses frame 1
                    GC.Collect();
                }
                JSonFile_FaceDetecion(JsonFileName);
            }
            else
            {
                bool CheckBreak = false;
                double PosFrame = 0;
                using (Emgu.CV.VideoCapture videoCapture = new Emgu.CV.VideoCapture(FullFileName))
                {
                    while (!CheckBreak)
                    {
                        for (int n = 0; n < DetectInterval; n++)
                        {
                            Emgu.CV.Mat VideoMat = videoCapture.QueryFrame();
                            PosFrame = videoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames);
                            if (VideoMat == null)
                            {
                                CheckBreak = true;
                                break;
                            }
                            else
                            {
                                CheckBreak = false;
                                if (n == 0)
                                {
                                    openCVResult = FaceDetect_Cascade(VideoMat, FaceFileName_Cascade);
                                    List<TSJSonClass> TSFaces = new List<TSJSonClass>();
                                    for (int i = 0; i < openCVResult.faces.Count; i++)
                                    {
                                        TSFaces.Add(new TSJSonClass { face = openCVResult.faces[i], enable = true });
                                        GC.Collect();
                                    }
                                    TSDic.Add(PosFrame.ToString("00000"), TSFaces);
                                    GC.Collect();
                                }
                            }
                        }
                    }
                    JSonFile_FaceDetecion(JsonFileName);
                }
            }
        }
        /// <summary>
        /// Face detection using YOLO classifier
        /// </summary>
        /// <param name="FullFileName"> Full input file name</param>
        /// <param name="JsonFileName"> Output JSon file name (saving faces information)</param>
        /// <param name="ConfigFile_YOLO"> YOLO config file (.cfg)</param>
        /// <param name="Weights_YOLO"> YOLO weight file (.weights)</param>
        /// <param name="Labels"> Label classes</param>
        /// <param name="DetectClass"> Classes wanted to be detected</param>
        /// <param name="DetectInterval"> Detect interval</param>
        public void FaceDetect(string FullFileName, string JsonFileName, string ConfigFile_YOLO, string Weights_YOLO, string[] Labels, int[] DetectClass, int DetectInterval)
        {
            if (File.Exists(JsonFileName))
                File.Delete(JsonFileName);
            string FileFormat = System.IO.Path.GetExtension(FullFileName);
            FileFormat = FileFormat.ToLower();
            OpenCVResult openCVResult = new OpenCVResult();
            TSDic.Clear();
            if (FileFormat == ".jpg")
            {
                using (Emgu.CV.Mat PictureMat = CvInvoke.Imread(FullFileName))
                {
                    openCVResult = FaceDetect_YOLO(PictureMat, ConfigFile_YOLO, Weights_YOLO, Labels, DetectClass);
                    List<TSJSonClass> TSFaces = new List<TSJSonClass>();
                    for (int i = 0; i < openCVResult.faces.Count; i++)
                    {
                        TSFaces.Add(new TSJSonClass { face = openCVResult.faces[i], enable = true });
                        GC.Collect();
                    }
                    TSDic.Add(1.ToString("00000"), TSFaces); //Image always uses frame 1
                    GC.Collect();
                }
                JSonFile_FaceDetecion(JsonFileName);
            }
            else
            {
                bool CheckBreak = false;
                double PosFrame = 0;
                using (Emgu.CV.VideoCapture videoCapture = new Emgu.CV.VideoCapture(FullFileName))
                {
                    while (!CheckBreak)
                    {
                        for (int n = 0; n < DetectInterval; n++)
                        {
                            Emgu.CV.Mat VideoMat = videoCapture.QueryFrame();
                            PosFrame = videoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames);
                            if (VideoMat == null)
                            {
                                CheckBreak = true;
                                break;
                            }
                            else
                            {
                                CheckBreak = false;
                                if (n == 0)
                                {
                                    openCVResult = FaceDetect_YOLO(VideoMat, ConfigFile_YOLO, Weights_YOLO, Labels, DetectClass);
                                    List<TSJSonClass> TSFaces = new List<TSJSonClass>();
                                    for (int i = 0; i < openCVResult.faces.Count; i++)
                                    {
                                        TSFaces.Add(new TSJSonClass { face = openCVResult.faces[i], enable = true });
                                        GC.Collect();
                                    }
                                    TSDic.Add(PosFrame.ToString("00000"), TSFaces);
                                    GC.Collect();
                                }
                            }
                        }
                    }
                    JSonFile_FaceDetecion(JsonFileName);
                }
            }
        }
        private OpenCVResult FaceDetect_Cascade(Emgu.CV.Mat Frame, string FaceFileName_Cascade)
        {
            long detectionTime;
            List<Rectangle> faces = new List<Rectangle>();
            List<Rectangle> eyes = new List<Rectangle>();

            DetectFace.DetectFaceOnly(
                Frame, FaceFileName_Cascade,
                faces,
                out detectionTime);

            OpenCVResult result = new OpenCVResult()
            {
                faces = faces,
                eyes = eyes,
            };

            return result;
        }
        private OpenCVResult FaceDetect_YOLO(Emgu.CV.Mat Frame, string ConfigFile_YOLO, string Weights_YOLO, string[] Label, int[] DetectClass)
        {
            List<Rectangle> faces = new List<Rectangle>();
            List<Rectangle> eyes = new List<Rectangle>();

            YOLOConfig_Parameter yOLOConfig_Parameter = new YOLOConfig_Parameter();
            yOLOConfig_Parameter = YOLOConfig_Parse(ConfigFile_YOLO);

            // Coverting between Emgu.CV Mat and OpenCVSharp Mat causes Tool exception
            // Using temp file to fix the problem
            if (File.Exists(@"Process.jpg"))
                File.Delete(@"Process.jpg");
            CvInvoke.Imwrite(@"Process.jpg", Frame);
            var org = Cv2.ImRead(@"Process.jpg");

            var w = org.Width;
            var h = org.Height;
            //setting blob, parameter are important
            //Size: Setting in config file
            var blob = CvDnn.BlobFromImage(org, 1 / 255.0, new OpenCvSharp.Size(yOLOConfig_Parameter.Width, yOLOConfig_Parameter.Height), new OpenCvSharp.Scalar(), true, false);
            var net = CvDnn.ReadNetFromDarknet(ConfigFile_YOLO, Weights_YOLO);
            net.SetInput(blob, "data");
            var prob = net.Forward();

            /* YOLO2 VOC output
             0 1 : center                    2 3 : w/h
             4 : confidence                  5 ~24 : class probability */
            const int prefix = 5;   //skip 0~4

            for (int i = 0; i < prob.Rows; i++)
            {
                var confidence = prob.At<float>(i, 4);
                if (confidence > yOLOConfig_Parameter.Threshold)
                {
                    //get classes probability
                    Cv2.MinMaxLoc(prob.Row[i].ColRange(prefix, prob.Cols), out _, out OpenCvSharp.Point max);
                    var classes = max.X;
                    var probability = prob.At<float>(i, classes + prefix);

                    if (probability != 0 && DetectClass.Contains(classes)) //more accuracy
                    {
                        //get center and width/height
                        var centerX = prob.At<float>(i, 0) * w;
                        var centerY = prob.At<float>(i, 1) * h;
                        var width = prob.At<float>(i, 2) * w;
                        var height = prob.At<float>(i, 3) * h;

                        var x1 = (centerX - width / 2) < 0 ? 0 : centerX - width / 2;
                        var y1 = (centerY - height / 2) < 0 ? 0 : centerY - height / 2;

                        faces.Add(new Rectangle(
                        (int)(x1),
                        (int)(y1),
                        (int)(width),
                        (int)(height)
                        ));
                    }
                }
            }
            net.Dispose();
            blob.Dispose();
            prob.Dispose();


            OpenCVResult result = new OpenCVResult()
            {
                eyes = eyes,
                faces = faces,
            };

            return result;
        }
        private void JSonFile_FaceDetecion(string OutFileName)
        {
            var json  = JsonConvert.SerializeObject(TSDic);
            using (StreamWriter sw = new StreamWriter(OutFileName))
            {
                sw.WriteAsync(json);
                System.Threading.Thread.Sleep(1000);
            }
            TSDic.Clear();
            GC.Collect();
        }
        /// <summary>
        /// Face reduction using faces information in JSon file
        /// </summary>
        /// <param name="FullFileName"> Full input file name</param>
        /// <param name="JSonFileName"> Input JSon file name</param>
        /// <param name="OutFileName"> Full output file name</param>
        public void FaceReduction(string FullFileName, string JSonFileName, string OutFileName)
        {
            if (!File.Exists(FullFileName) || !File.Exists(JSonFileName))
                return;
            JSon_Parser(JSonFileName);
            List<Rectangle> faces = new List<Rectangle>();
            string FileFormat = System.IO.Path.GetExtension(FullFileName);
            FileFormat = FileFormat.ToLower();
            if (FileFormat == ".jpg")
            {
                using (Emgu.CV.Mat PictureMat = CvInvoke.Imread(FullFileName))
                {
                    if (TSDic.ContainsKey(1.ToString("00000")))
                    {
                        faces.Clear();
                        foreach (var FrameFaces in TSDic[1.ToString("00000")])
                        {
                            if (FrameFaces.enable == false)
                                faces.Add(FrameFaces.face);
                        }
                    }

                    if (PictureMat != null)
                    {
                        //********** Face Reduction here **********//
                        for (int i = 0; i < faces.Count; i++)
                            CvInvoke.Rectangle(PictureMat, faces[i], new Emgu.CV.Structure.Bgr(Color.Black).MCvScalar, -1);
                    }
                    CvInvoke.Imwrite(OutFileName, PictureMat);
                }
            }
            else
            {
                bool CheckBreak = false;
                double PosFrame = 0;
                using (Emgu.CV.VideoCapture objVideoCapture = new Emgu.CV.VideoCapture(FullFileName))
                {
                    Emgu.CV.VideoWriter VW = new Emgu.CV.VideoWriter(OutFileName, Emgu.CV.VideoWriter.Fourcc('M', 'P', '4', 'V'), (int)objVideoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps), new System.Drawing.Size((int)objVideoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth), (int)objVideoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight)), true);
                    while (!CheckBreak)
                    {
                        Emgu.CV.Mat objMat = objVideoCapture.QueryFrame();
                        PosFrame = objVideoCapture.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames);

                        //Changing faces list every detect interval
                        if (TSDic.ContainsKey(PosFrame.ToString("00000")))
                        {
                            faces.Clear();
                            foreach (var FrameFaces in TSDic[PosFrame.ToString("00000")])
                            {
                                if (FrameFaces.enable == false)
                                    faces.Add(FrameFaces.face);
                            }
                        }

                        if (objMat == null)
                        {
                            CheckBreak = true;
                            break;
                        }
                        else
                        {
                            CheckBreak = false;
                            //********** Face Reduction here **********//
                            for (int i = 0; i < faces.Count; i++)
                                CvInvoke.Rectangle(objMat, faces[i], new Emgu.CV.Structure.Bgr(Color.Black).MCvScalar, -1);
                        }
                        VW.Write(objMat);
                        objMat.Dispose();
                        GC.Collect();
                    }
                    VW.Dispose();
                }
            }
        }
        private class OpenCVResult
        {
            public List<Rectangle> faces { get; set; }
            public List<Rectangle> eyes { get; set; }
        }
        private class YOLOConfig_Parameter
        {
            public double Width;
            public double Height;
            public double Threshold;
        }
        private YOLOConfig_Parameter YOLOConfig_Parse(string ConfigFile_YOLO)
        {
            string line;
            YOLOConfig_Parameter yOLOConfig_Parameter = new YOLOConfig_Parameter();
            StreamReader SR = new StreamReader(ConfigFile_YOLO);
            while ((line = SR.ReadLine()) != null)
            {
                if (line.Contains("width"))
                    yOLOConfig_Parameter.Width = Convert.ToDouble(line.Substring(line.IndexOf("=") + 1, line.Length - line.IndexOf("=") - 1)); //Example: width=416
                else if (line.Contains("height"))
                    yOLOConfig_Parameter.Height = Convert.ToDouble(line.Substring(line.IndexOf("=") + 1, line.Length - line.IndexOf("=") - 1)); //Examle: height=416
                else if (line.Contains("thresh"))
                    yOLOConfig_Parameter.Threshold = Convert.ToDouble(line.Substring(line.IndexOf("=") + 1, line.Length - line.IndexOf("=") - 1));  //Example: thresh=0.6 *Do not write thresh=.6
                else
                    continue;
            }
            SR.Close();
            SR.Dispose();
            return yOLOConfig_Parameter;
        }
        private void JSon_Parser(string JSonFileName)
        {
            List<TSJSonClass> JSon_Faces = new List<TSJSonClass>();
            using (StreamReader sr = new StreamReader(JSonFileName))
            {
                string json = sr.ReadToEnd();
                System.Threading.Thread.Sleep(1000);
                TSDic.Clear();
                TSDic = JsonConvert.DeserializeObject<Dictionary<string, List<TSJSonClass>>>(json);
            }
        }

        public class TSJSonClass
        {
            public Rectangle face = new Rectangle();
            public bool enable = true;
        }
        public Dictionary<string, List<TSJSonClass>> TSDic = new Dictionary<string, List<TSJSonClass>>();
    }
}
