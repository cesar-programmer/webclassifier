/* eslint-disable no-undef */
let net
let webcam
const classifier = knnClassifier.create()
const webcamElement = document.getElementById('webcam')
const notification = document.getElementById('notification')

async function app () {
  console.log('Cargando modelo de identificación de imágenes')
  net = await mobilenet.load()
  console.log('Carga terminada')

  // Obtenemos datos de la webcam
  webcam = await tf.data.webcam(webcamElement)

  // Procesamos las imágenes capturadas de la webcam en tiempo real
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

    const classes = ['Untrained', 'Casco Detectado', 'Sin Casco', 'OK', 'Otro']

    // Mostrar la predicción del clasificador
    try {
      const label = classes[result2.label]
      const probability = result2.confidences[result2.label] || 0
      document.getElementById('console2').innerText = `
        Predicción: ${label}\n
        Probabilidad: ${(probability * 100).toFixed(2)}%
      `

      // Mostrar notificación en pantalla
      if (label === 'Casco Detectado') {
        showNotification('Casco Detectado', 'success')
      } else if (label === 'Sin Casco') {
        showNotification('Sin Casco', 'error')
      } else {
        hideNotification()
      }
    } catch (error) {
      document.getElementById('console2').innerText = 'Modelo no entrenado'
      hideNotification()
    }

    // Liberar el tensor para liberar memoria
    img.dispose()

    // Esperar el próximo frame
    await tf.nextFrame()
  }
}

// Función para añadir ejemplos de entrenamiento al clasificador
async function addExample (classId) {
  const img = await webcam.capture()
  const activation = net.infer(img, true)
  classifier.addExample(activation, classId)
  img.dispose()
}

// Función para mostrar notificaciones
function showNotification (message, type) {
  notification.innerText = message
  notification.className = type
}

// Función para ocultar notificaciones
function hideNotification () {
  notification.className = 'hidden'
}

// Función para almacenar el clasificador en el localStorage
const saveKnn = async () => {
  const strClassifier = JSON.stringify(
    Object.entries(classifier.getClassifierDataset()).map(([label, data]) => [
      label,
      Array.from(data.dataSync()),
      data.shape
    ])
  )
  const storageKey = 'knnClassifier'
  localStorage.setItem(storageKey, strClassifier)
}

// Función para recuperar el clasificador del localStorage
const loadKnn = async () => {
  const storageKey = 'knnClassifier'
  const datasetJson = localStorage.getItem(storageKey)
  classifier.setClassifierDataset(
    Object.fromEntries(
      JSON.parse(datasetJson).map(([label, data, shape]) => [
        label,
        tf.tensor(data, shape)
      ])
    )
  )
}

app()
