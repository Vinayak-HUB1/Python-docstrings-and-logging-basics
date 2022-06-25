from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os


gate = "AND_GATE"
log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "running_logs.log"),    level=logging.INFO,
                    filemode='a',
                    format='[%(asctime)s:%(levelname)s:%(module)s]:%(message)s')


def main(data, modelName, plotName, eta, epochs):
    df = pd.DataFrame(data)
    X, y = prepare_data(df)
    logging.info(f"This is raw data:\n {df}")

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df, model, filename=plotName)

if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(f">>>>>>>>> starting training for {gate} >>>>>>>>>")
        main(data=AND, modelName="and.model",
             plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(
            f"<<<<<<<<< training is completed for {gate} <<<<<<<<< \n\n")
    except Exception as e:
        logging.error(e)
        raise e
