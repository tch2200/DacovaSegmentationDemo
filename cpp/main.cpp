#include <tuple>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

using namespace std;

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const int CHANNELS = 3;
const int PADDING_COLOR = 0;
const int BACKGROUND_CLASS_ID = 0;

struct Resize
{
    cv::Mat resized_image;
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

ov::CompiledModel prepareModel(const std::string &modeFileName, const std::string &deviceName = "CPU")
{
    // Step 1. Initialize OpenVINO Runtime core
    ov::Core core;
    // Step 2. Read a model
    std::shared_ptr<ov::Model> model = core.read_model(modeFileName);

    if (model->get_parameters().size() != 1)
    {
        throw std::logic_error("Segment model must have only one input");
    }
    // cout << "Number of input: " << model->get_parameters().size() << endl;
    // cout << "----------------------------" << endl;
    model->reshape({1, INPUT_HEIGHT, INPUT_WIDTH, CHANNELS}); // change input from ?*HEIGHT*WIDTH*3 to 1*HEIGHT*WIDTH*3

    // Step 3. Inizialize Preprocessing for the model
    ov::preprocess::PrePostProcessor ppp(model);

    // Specify input image format:
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);

    // Specify preprocess pipeline to input image without resizing
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale({255., 255., 255.})
        .mean({0.485, 0.456, 0.406})
        .scale({0.229, 0.224, 0.225});

    //  Specify model's input layout
    ppp.input()
        .model()
        .set_layout("NHWC");

    ppp.output()
        .tensor()
        .set_element_type(ov::element::f32);    

    // Embed above steps in the graph
    model = ppp.build();
    ov::CompiledModel compiledModel = core.compile_model(model, deviceName);
    return compiledModel;
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
    cv::Mat croppedImg;

    // crop roi
    croppedImg = cropWithRoi(img, roi);

    // resize
    Resize resize = resizeAndPad(croppedImg, target_size);

    return resize;
}

void printOutput(ov::Tensor output)
{
    cout << "Output shape: " << output.get_shape();
    cout << "Output type: " << output.get_element_type() << endl;
    // cout << output.data<float>()[0] << endl;
    auto data = output.data<float>();

    for (auto i = 0; i < 10; i++)
    {
        cout << data[i] << " ";
    }
    cout << endl;
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
    cv::imwrite("../../examples/output_openvino_cpp/demo_cpp.jpg", showImg);
    cout << "Save predicted image path: examples/output_openvino_cpp/demo_cpp.jpg" << endl;
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

        CocoResult result(
            -1,                 // int imageId
            classNames[i + 1],  // string category
            i + 1,              // int categoryId
            boxes,              // vector<cv::Rect> boxes
            contours_poly);     // vector<vector<cv::Point>> segmentations
        results.push_back(result);
    }
    return results;
}

vector<CocoResult> postProcessing(
    ov::Tensor output,
    cv::Mat &oriImg,
    vector<float> &list_prob_threshold,
    Resize resize,
    cv::Size originImgShape,
    Roi roi,
    vector<string> &classNames)
{
    auto outShape = output.get_shape(); // NHWC
    auto outHeight = outShape[1];
    auto outWidth = outShape[2];
    auto outChannel = outShape[3];

    const float *data = output.data<float>(); // NHWC
    cv::Mat mask = cv::Mat(outHeight, outWidth, CV_8UC1);

    for (int rowId = 0; rowId < outHeight; rowId++)
    {
        for (int colId = 0; colId < outWidth; colId++)
        {
            int classId = 0;
            float maxProb = -1.0f;
            for (int channelId = 0; channelId < outChannel; channelId++)
            {

                int index_pixel = (rowId * outWidth + colId) * outChannel + channelId;
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
    std::string modeFileName = "../../weights/openvino/model_20221222_13530_576984.xml";
    std::string img_path = "../../examples/imgs/demo.jpg";    
    // init config
    Roi roi = {0, 0, 11900, 11900};
    cv::Size targetSize = cv::Size(INPUT_WIDTH, INPUT_HEIGHT);
    vector<float> list_prob_threshold = {0.5, 0.6, 0.4, 0.5};
    vector<string> classNames = {"background", "hole", "kizu", "yolore"};

    // init model
    ov::CompiledModel compiledModel = prepareModel(modeFileName);
    ov::InferRequest req = compiledModel.create_infer_request();

    // start count time
    using timer = std::chrono::high_resolution_clock;
    timer::time_point lastTime = timer::now();

    // init input data
    cv::Mat img = cv::imread(img_path);
    Resize resize = preprocess(img, roi, targetSize);
    cv::Mat processedImg = resize.resized_image;
    float *input_data = (float *)processedImg.data;
    ov::Tensor input_tensor = ov::Tensor(compiledModel.input().get_element_type(), compiledModel.input().get_shape(), input_data);

    // infer
    req.set_input_tensor(input_tensor);
    req.start_async();
    req.wait();
    auto output = req.get_output_tensor();

    // post processing
    vector<CocoResult> results = postProcessing(
        output,
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