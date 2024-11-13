/* eslint-disable no-undef */
let net
let webcam
const classifier = knnClassifier.create()
const webcamElement = document.getElementById('webcam')

async function app () {
  console.log('Cargando modelo de identificación de imágenes')
  net = await mobilenet.load()
  console.log('Carga terminada')

  // Obtenemos datos de la webcam
  webcam = await tf.data.webcam(webcamElement)

  // Procesamos las imagenes capturadas de la webcam en tiempo real
  while (true) {
    const img = await webcam.capture()

    // Realizar la inferencia y predecir con el clasificador
    const activation = net.infer(img, 'conv_preds')
    let result2
    try {
      result2 = await classifier.predictClass(activation)
    } catch (error) {
      result2 = {}
    }

    const classes = ['Untrained', 'pikachu', 'bulbasaur', 'cesar', 'OK', 'Rock']

    // Mostrar la predicción del clasificador `
    try {
      document.getElementById('console2').innerText = `
        prediction: ${classes[result2.label]}\n
        probability: ${result2.confidences[result2.label]}
      `
    } catch (error) {
      document.getElementById('console2').innerText = 'Untrained'
    }

    // Liberar el tensor para liberar memoria
    img.dispose()

    // Esperar el proximo frame
    await tf.nextFrame()
  }
}

// Funcion para anadir ejemplos de entrenamiento al clasificador
async function addExample (classId) {
  const img = await webcam.capture()
  const activation = net.infer(img, true)
  classifier.addExample(activation, classId)
  img.dispose()
}

// Funcion para almacenar el clasificador en el localStorage
const saveKnn = async () => {
  const strClassifier = JSON.stringify(Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [label, Array.from(data.dataSync()), data.shape]))
  const storageKey = 'knnClassifier'
  localStorage.setItem(storageKey, strClassifier)
}

// Funcion para recuperar el clasificador del localStorage
const loadKnn = async () => {
  const storageKey = 'knnClassifier'
  const datasetJson = localStorage.getItem(storageKey)
  classifier.setClassifierDataset(Object.fromEntries(JSON.parse(datasetJson).map(([label, data, shape]) => [label, tf.tensor(data, shape)])))
}

app()
