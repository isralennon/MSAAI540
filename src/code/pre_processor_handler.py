import json
import logging

logging.basicConfig(level=logging.INFO)

def top_level_attributes(obj):
    if isinstance(obj, dict):
        return list(obj.keys())
    else:
        return [attr for attr in dir(obj)
                if not attr.startswith("_") and not callable(getattr(obj, attr))]

def preprocess_handler(inference_record):
    """
        Sample inference_record record: {'eventVersion': '0', 'groundTruthData': {'data': '136.48', 'encoding': 'CSV'}, 'captureData': {'endpointInput': {'encoding': 'CSV', 'data': '13.095,29.854,1100.1,4.4058,0.0,0.0,0.0,1009.5,0.0,1008.9', 'observedContentType': 'text/csv'}, 'endpointOutput': {'encoding': 'CSV', 'data': '147.81324768066406\n', 'observedContentType': 'text/csv; charset=utf-8'}}, 'eventMetadata': {'eventId': '439cc9e4-3bb2-4724-9bbb-e2f92f16dc26', 'inferenceId': '98', 'inferenceTime': '2026-02-22T02:48:50Z'}}
    """
    # logging.info(f"Raw record: {inference_record}")
    # if hasattr(inference_record, 'ground_truth'):
    #     value = getattr(inference_record, 'ground_truth')
    #     logging.info(value)
    # else:
    #     logging.info("Attribute 'ground_truth' does not exist.")
    # logging.info(f"Raw record: {top_level_attributes(inference_record)}")

    try:
        prediction = float(inference_record.endpoint_output.data.strip())
        groundTruthData = float(inference_record.ground_truth.data.strip())
    except Exception as e:
        logging.error(f"Error parsing output: {e}")
        prediction = None
        groundTruthData = None

    # logging.info(f"Processed prediction: {prediction}, ground truth: {groundTruthData}")

    return {
        "endpointOutput_prediction": prediction,
        "groundTruthData_0": groundTruthData
    }