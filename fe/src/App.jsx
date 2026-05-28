import React, { useState } from 'react'
import VI_MAP from './viMap'

export default function App() {
  const [file, setFile] = useState(null)
  const [annotated, setAnnotated] = useState(null)
  const [detections, setDetections] = useState([])
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) return
    setLoading(true)
    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/predict/image', {
        method: 'POST',
        body: form,
      })
      const data = await res.json()
      setDetections(data.detections || [])
      if (data.annotated_image_b64) {
        setAnnotated('data:image/jpeg;base64,' + data.annotated_image_b64)
      }
    } catch (err) {
      console.error(err)
      alert('Error calling API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>VN Traffic Sign — Image Inference</h1>
      <form onSubmit={handleSubmit} className="form">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Predicting...' : 'Predict Image'}
        </button>
      </form>

      <div className="results">
        {annotated && (
          <div>
            <h2>Annotated</h2>
            <img src={annotated} alt="annotated" className="annotated" />
          </div>
        )}

        <div>
          <h2>Detections</h2>
          <ul>
            {detections.length === 0 && <li>No detections</li>}
            {detections.map((d, i) => {
              const vi = VI_MAP[d.class_name] || VI_MAP[d.class_id] || d.class_name
              return (
                <li key={i}>
                  <strong>{vi}</strong> — {d.class_name} (id: {d.class_id}) — conf: {d.confidence.toFixed(3)} — bbox: [{d.bbox.map(v=>Math.round(v)).join(', ')}]
                </li>
              )
            })}
          </ul>
        </div>
      </div>
    </div>
  )
}
