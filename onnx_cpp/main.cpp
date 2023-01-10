#include <tuple>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

using namespace std;

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const int CHANNELS = 3;
const int PADDING_COLOR = 0;
const int BACKGROUND_CLASS_ID = 0;

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

struct Resize
{
    cv::Mat resized_image;
    std::vector<float> data;
    int dw;
    int dh;
    int ori_w;
    int ori_h;
};

struct Roi
{
    int xmin, ymin, xmax, ymax;
};

struct CocoResult
{
    int imageId;
    string category;
    int categoryId;
    vector<cv::Rect> boxes;
    vector<vector<cv::Point>> segmentations;

    CocoResult(
        int imageId, string category, int categoryId,
        vector<cv::Rect> boxes, vector<vector<cv::Point>> segmentations)
        : imageId(imageId),
          category(category),
          categoryId(categoryId),
          boxes(boxes),
          segmentations(segmentations) {}

    CocoResult()
        : imageId(-1),
          category("background"),
          categoryId(0),
          boxes({}),
          segmentations({}) {}
};

// utils
void printImg(const cv::Mat &img)
{
    cout << "Shape: " << img.rows << endl;
    cout << "cv::Mat img = " << endl
         << " " << img << endl
         << endl;
}

void saveFile(const string &filename, const cv::Mat &img)
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "Result" << img;
    fs.release();
}

cv::Size getSizeImg(const cv::Mat &img)
{
    const size_t channels = img.channels();
    const size_t height = img.rows;
    const size_t width = img.cols;
    cv::Size sizeImg = cv::Size(width, height);
    return sizeImg;
}


cv::Mat cropWithRoi(const cv::Mat &img, const Roi roi)
{

    cv::Mat croppedImg;

    cv::Size sizeImg = getSizeImg(img);
    int xmin, ymin, xmax, ymax, width_rect, height_rect;

    xmin = int(roi.xmin);
    ymin = int(roi.ymin);
    xmax = int(roi.xmax);
    ymax = int(roi.ymax);

    xmin = max(0, xmin);
    ymin = max(0, ymin);
    xmax = min(sizeImg.width, xmax);
    ymax = min(sizeImg.height, ymax);
    width_rect = xmax - xmin;
    height_rect = ymax - ymin;

    cv::Rect myRoi(xmin, ymin, width_rect, height_rect);
    img(myRoi).copyTo(croppedImg);
    return croppedImg;
}

Resize resizeAndPad(const cv::Mat &img, const cv::Size new_shape)
{
    float width = img.cols;
    float height = img.rows;
    float r = float(new_shape.width / max(width, height));
    int new_unpadW = int(round(width * r));
    int new_unpadH = int(round(height * r));
    Resize resize;
    cv::resize(img, resize.resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);

    resize.dw = new_shape.width - new_unpadW;
    resize.dh = new_shape.height - new_unpadH;
    resize.ori_w = width;
    resize.ori_h = height;
    cv::Scalar color = cv::Scalar(PADDING_COLOR, PADDING_COLOR, PADDING_COLOR);
    cv::copyMakeBorder(resize.resized_image, resize.resized_image, 0, resize.dh,
                       0, resize.dw, cv::BORDER_CONSTANT, color);
    return resize;
}

Resize preprocess(const cv::Mat &img, Roi roi, cv::Size target_size)
{
    // crop roi
    cv::Mat croppedImg;
    croppedImg = cropWithRoi(img, roi);

    // resize
    Resize resize = resizeAndPad(croppedImg, target_size);

    // cv::Mat rgbImg;
    cv::cvtColor(resize.resized_image, resize.resized_image, cv::COLOR_BGR2RGB);

    resize.resized_image.convertTo(resize.resized_image, CV_32FC3, 1/255.);  //  /255
    cv::Mat channels[3];
    cv::split(resize.resized_image, channels);
    
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resize.resized_image);    

    return resize;
}

cv::Mat getMaskFromClassId(const cv::Mat &mask, const int classId)
{
    cv::Size size = mask.size();
    cv::Mat outMask = mask.clone();
    // at (int row, int col)
    for (int32_t rowId(0); rowId < mask.rows; rowId++)
    {
        for (int32_t colId(0); colId < mask.cols; colId++)
        {
            if (mask.at<int8_t>(rowId, colId) == classId)
            {
                outMask.at<int8_t>(rowId, colId) = 255;
            }
            else
            {
                outMask.at<int8_t>(rowId, colId) = 0;
            }
        }
    }

    return outMask;
}

void visualize(const cv::Mat &oriImg, const vector<CocoResult> &results)
{
    cv::Mat showImg = oriImg.clone();
    CocoResult result;
    vector<vector<cv::Point>> segmentations;
    vector<cv::Rect> boxes;
    string category;
    cv::Mat drawImg;
    cv::Scalar color;

    for (int k = 0; k < results.size(); k++)
    {
        result = results[k];
        segmentations = result.segmentations;
        boxes = result.boxes;
        category = result.category;
        color = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);

        for (int i = 0; i < boxes.size(); i++)
        {
            drawImg = cv::Mat(oriImg.size(), CV_8UC3, CV_RGB(0, 0, 0));
            cv::fillPoly(drawImg, segmentations[i], color);
            cv::addWeighted(showImg, 0.8, drawImg, 0.2, 0, showImg);
            cv::rectangle(showImg, boxes[i].tl(), boxes[i].br(), color, 1);
            cv::putText(showImg, category, boxes[i].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 1);
        }
    }
    cv::imwrite("../../examples/output_onnx_cpp/demo_cpp.jpg", showImg);
    cout << "Save result at ../../examples/output_onnx_cpp/demo_cpp.jpg" << endl;
}

vector<CocoResult> convertMaskToCocoFormat(
    const cv::Mat &oriImg,
    const cv::Mat &oriMask,
    const vector<string> &classNames,
    const int thresholdArea = 25,
    const int thresholdWidth = 10,
    const int thresholdHeight = 10)
{
    vector<CocoResult> results;
    vector<cv::Mat> mask_list;
    for (auto classId = 1; classId < classNames.size(); classId++)
    {
        cv::Mat classMask = getMaskFromClassId(oriMask, classId);
        mask_list.push_back(classMask);        
    }

    for (int i = 0; i < mask_list.size(); i++)
    {
        cv::Mat mask = mask_list[i];
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat contourOutputs = mask.clone();
        cv::findContours(contourOutputs, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        vector<vector<cv::Point>> contours_poly;
        vector<cv::Rect> boxes; // Rect(x, y, w, h)
        double peri, area;
        for (size_t i = 0; i < contours.size(); i++)
        {
            peri = cv::arcLength(contours[i], true);
            vector<cv::Point> contour_poly;
            cv::approxPolyDP(contours[i], contour_poly, 0.02 * peri, true);

            area = cv::contourArea(contours[i]);
            if (area < thresholdArea)
                continue;

            cv::Rect box = cv::boundingRect(contour_poly);
            if (box.width < thresholdWidth || box.height < thresholdHeight)
                continue;
            
            contours_poly.push_back(contour_poly);
            boxes.push_back(box);
        }
        if (contours_poly.size() == 0)
            continue;

        // int imageId;
        // string category;
        // int categoryId;
        // vector<cv::Rect> boxes;
        // vector<vector<cv::Point>> segmentations;
        CocoResult result(
            -1,
            classNames[i + 1],
            i + 1,
            boxes,
            contours_poly);
        results.push_back(result);
    }

    return results;
}
size_t vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

vector<CocoResult> postProcessing(
    vector<float> output,
    vector<int64_t> outputDims,
    cv::Mat &oriImg,
    vector<float> &list_prob_threshold,
    Resize resize,
    cv::Size originImgShape,
    Roi roi,
    vector<string> &classNames)
{
    const float *data = output.data(); // NHWC
    int outHeight = outputDims.at(1);
    int outWidth = outputDims.at(2);
    int outChannel = outputDims.at(3);
    
    cv::Mat mask = cv::Mat(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC1);
    for (int rowId = 0; rowId < outHeight; rowId++)
    {
        for (int colId = 0; colId < outWidth; colId++)
        {
            int classId = 0;
            float maxProb = -1.0f;
            for (int channelId = 0; channelId < outChannel; channelId++)
            {

                int index_pixel = (rowId * outWidth + colId) * outChannel + channelId;
                if (rowId == 0 && channelId == 0 && colId < 10) {
                    cout << data[index_pixel] << " ";
                }
                float prob = data[index_pixel];

                if (prob > maxProb)
                {
                    classId = channelId;
                    maxProb = prob;
                }
            }

            if (classId != BACKGROUND_CLASS_ID && maxProb > list_prob_threshold[classId])
            {
                mask.at<uint8_t>(rowId, colId) = classId;
            }
            else
            {
                mask.at<uint8_t>(rowId, colId) = BACKGROUND_CLASS_ID;
            }
        }
    }
    saveFile("mask.yml", mask);

    cv::Rect notPadRoi(0, 0, outWidth - resize.dh, outHeight - resize.dw);
    cv::Mat notPadMask;
    mask(notPadRoi).copyTo(notPadMask);

    cv::resize(
        notPadMask,
        notPadMask,
        cv::Size(resize.ori_w, resize.ori_h),
        0,
        0,
        cv::INTER_NEAREST);

    cv::Mat oriMask = cv::Mat::zeros(originImgShape.height, originImgShape.width, CV_8UC1);

    // edit roi
    int xmax = min(originImgShape.width, roi.xmax);
    int ymax = min(originImgShape.height, roi.ymax);
    cv::Mat subMat = oriMask(cv::Rect(roi.xmin, roi.ymin, xmax - roi.xmin, ymax - roi.ymin));
    notPadMask.copyTo(subMat);    

    vector<CocoResult> results = convertMaskToCocoFormat(oriImg, oriMask, classNames);

    return results;
}

int main()
{
    std::string modelFilePath = "../../weights/onnx/model_2023110_191736_351410.onnx";
    std::string img_path = "../../examples/imgs/demo.jpg";

    // init config
    Roi roi = {0, 0, 11900, 11900};
    cv::Size targetSize = cv::Size(INPUT_WIDTH, INPUT_HEIGHT);
    vector<float> list_prob_threshold = {0.1, 0.6, 0.4, 0.1};
    vector<string> classNames = {"background", "hole", "kizu", "yogore"};

    // init model
    const int batchSize = 1;
    string instanceName{"segment-onnx"};
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                instanceName.c_str());
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, modelFilePath.c_str(), sessionOptions);
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();
    
    // Set batch size to 1
    const char* inputName = session.GetInputName(0, allocator);
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    if (inputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting input batch size to "
                  << batchSize << "." << std::endl;
        inputDims.at(0) = batchSize;
    }

    const char* outputName = session.GetOutputName(0, allocator);
    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    if (outputDims.at(0) == -1)
    {
        std::cout << "Got dynamic batch size. Setting output batch size to "
                  << batchSize << "." << std::endl;
        outputDims.at(0) = batchSize;
    }

    // start count time
    using timer = std::chrono::high_resolution_clock;
    timer::time_point lastTime = timer::now();

    size_t inputTensorSize = vectorProduct(inputDims);    
    vector<const char*> inputNames{inputName};
    vector<Ort::Value> inputTensors;

    size_t outputTensorSize = vectorProduct(outputDims);
    vector<float> outputTensorValues(outputTensorSize);
    vector<const char*> outputNames{outputName};
    vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator,
        OrtMemType::OrtMemTypeDefault
    );
    
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo,
        outputTensorValues.data(),
        outputTensorSize,
        outputDims.data(),
        outputDims.size()
    ));
    
    // Read + preprocessing input image
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);   //HWC - BGR - UINT8
    Resize resize = preprocess(img, roi, targetSize);
    cv::Mat processedImg = resize.resized_image;
    
    float *inputTensorValues = (float* )processedImg.data;    
        
    inputTensors.emplace_back(
        Ort::Value::CreateTensor<float>(
            memoryInfo,            
            inputTensorValues,
            inputTensorSize,
            inputDims.data(),
            inputDims.size())
    );
    // inference
    session.Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputTensors.data(), 
        numInputNodes /*Num of inputs*/,
        outputNames.data(),
        outputTensors.data(), 
        numOutputNodes /*Num of outputs*/
    );
    
    // post processing    
    vector<CocoResult> results = postProcessing(
        outputTensorValues,
        outputDims,
        img,
        list_prob_threshold,
        resize,
        img.size(),
        roi,
        classNames);

    auto currTime = timer::now();
    auto timeInfer = (currTime - lastTime);
    cout << "Time forward: " << std::chrono::duration_cast<std::chrono::milliseconds>(timeInfer).count() << "ms" << endl;

    // visualize
    visualize(img, results);

    return 0;
}