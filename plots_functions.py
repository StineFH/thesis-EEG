import matplotlib.pyplot as plt
import torch



abs_pred_error = torch.tensor([34, 341, 19, 19, 29, 29, 101, 349, 10, 13])

def abs_prediction_error(abs_pred_error):
    """
    X axis is the distance in points to last know point. 
    The first half and last half of points are displayed where the last half 
    has reversed points to fit with distance to points after. 
    """
    assert len(abs_pred_error) % 2 == 0
    
    half_len_preds = int(len(abs_pred_error)/2)
    dist_from_known = range(1, half_len_preds+1) 
    
    plt.plot(dist_from_known, abs_pred_error[:half_len_preds], label='First', color = 'b')
    plt.plot(dist_from_known, list(abs_pred_error[half_len_preds:])[::-1], label='Last', color = 'g')
    plt.title('Absolute prediction error')
    plt.xlabel('Distance from known point')
    plt.xticks(dist_from_known)
    plt.ylabel('Absolute prediction error')
    plt.legend()
    plt.show()
    
abs_prediction_error(abs_pred_error)

train_loss = [6748.8134765625,7367.173828125,4344.22216796875,
               2343.082763671875]
val_loss = [742.7548217773438,773.6451416015625, 823.059814453125,
             796.8303833007812]

def val_train_loss(val_loss, train_loss, x_axis = 'Epochs'):
    assert len(val_loss) == len(train_loss)
    epochs = range(1, len(val_loss)+1)
    
    plt.plot(epochs, val_loss, label='Validation', color = 'b')
    plt.plot(epochs, train_loss, label='Train', color = 'g')
    plt.title('Loss over  ' + x_axis)
    plt.xlabel(x_axis)
    plt.xticks(epochs)
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()
    
val_train_loss(val_loss, train_loss, 'Batch')
