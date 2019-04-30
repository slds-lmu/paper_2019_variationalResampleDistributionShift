set cuda = False in utils/BBBlayers.py to run cpu version
using gpu, 1 epoch(around 1800 batches for the mnist) take 10 minutes, while using cpu takes 2 hours
using gpu, 1 epoch of cv on clustered data take 1 hour (approximately 5  epoch without cv)
| Validation Epoch #6			Loss: 1.6757 Acc@1: 46.00%
| Elapsed time : 0:59:09

