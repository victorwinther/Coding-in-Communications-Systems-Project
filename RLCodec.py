import numpy as np

def RLenc(input,G):
## Run length encoder
# input: input sequence
# G: maximal run length handled

    input = input.flatten()
    encoded = np.nan*np.ones((len(input),), dtype=float)
    idxInput = 0
    idxOutput = 0
    lengthNaN = G

    for loop in np.arange(len(input)):
        currSymb = input[idxInput]
        currRunLength = np.nonzero(input[idxInput:]!=currSymb)[0]
        if not currRunLength.size:
            currRunLength = np.array([len(input)-idxInput])

        if currRunLength[0] < lengthNaN:
            encoded[idxOutput] = currRunLength[0]
            idxInput = idxInput + currRunLength[0]
        else:
            encoded[idxOutput] = np.nan
            idxInput = idxInput + lengthNaN

        idxOutput = idxOutput + 1

        if currRunLength[0] == lengthNaN:
            encoded[idxOutput] = 0
            idxOutput = idxOutput + 1
        
        if idxInput >= len(input):
            break

    if idxOutput < len(input):
        RLEnc = encoded[0:idxOutput]

    if(input[0]):
        RLEnc = np.insert(RLEnc, 0, 0)

    return RLEnc

def RLdec(encoded, G, rowLength):
##  Run length decoder
# encoded: sequence to decode 
# G: maximal run length handled
# rowLength: length of a row for reconstruction nto a matrix
# decoded: output as 2D np array with row length rowLength

    idxDec = 0
    currValue = 0

    for idxIn in np.arange(len(encoded)):
        decodedTemp = np.full((int(encoded[idxIn]),),currValue)
        idxDec = idxDec+encoded[idxIn]
        if encoded[idxIn]!=G:
            currValue = 1-currValue
        
        try:
            decoded
        except:
            decoded = decodedTemp
        else:
            decoded = np.append(decoded, decodedTemp)

    decoded = np.reshape(decoded, (int(decoded.size/rowLength), rowLength))

    return decoded


