import React, { useRef, useState, useEffect } from 'react'
import VI_MAP from './viMap'

export default function App() {
  const [tab, setTab] = useState('image')
  const [file, setFile] = useState(null)
  const [annotated, setAnnotated] = useState(null)
  const [detections, setDetections] = useState([])
  const [loading, setLoading] = useState(false)
  const [videoAnnotated, setVideoAnnotated] = useState(null)
  const [videoFrames, setVideoFrames] = useState(null)
  const [videoStreamSrc, setVideoStreamSrc] = useState(null)
  const [videoStatus, setVideoStatus] = useState('')
  const [videoFps, setVideoFps] = useState(null)
  const [detectHistory, setDetectHistory] = useState([])
  const lastFrameTimeRef = useRef(null)
  const previewRef = useRef(null)
  const [isFs, setIsFs] = useState(false)

  useEffect(() => {
    function onFsChange() {
      setIsFs(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  function pushDebug(message) {
    const line = `${new Date().toLocaleTimeString()} ${message}`
    console.log(line)
  }

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

  async function handleVideoSubmit(e) {
    e.preventDefault()
    if (!file) return
    pushDebug(`Video submit clicked: name=${file.name}, size=${file.size}`)
    setLoading(true)
    setVideoAnnotated(null)
    setVideoStreamSrc(null)
    setVideoFrames(null)
    setVideoFps(null)
    setVideoStatus('Connecting to server...')
    setDetectHistory([])
    lastFrameTimeRef.current = null

    try {
      setVideoStatus('Uploading video via HTTP...')
      const form = new FormData()
      form.append('file', file)
      const uploadUrl = 'http://localhost:8000/upload/video'
      pushDebug(`Uploading via fetch: ${uploadUrl}`)
      const uploadRes = await fetch(uploadUrl, { method: 'POST', body: form })
      const uploadData = await uploadRes.json()
      if (!uploadRes.ok) {
        throw new Error(uploadData?.detail || uploadData?.message || 'Upload failed')
      }

      const jobId = uploadData.job_id
      pushDebug(`Upload complete, job_id=${jobId}`)
      setVideoStatus('Video uploaded. Opening stream...')

      const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000/ws/video_stream/${jobId}`
      pushDebug(`Connecting websocket: ${wsUrl}`)
      const ws = new WebSocket(wsUrl)
      ws.binaryType = 'arraybuffer'
      let lastBlobUrl = null

      ws.onopen = () => {
        pushDebug('WebSocket open')
        setVideoStatus('Processing frames...')
      }

      ws.onmessage = (evt) => {
        try {
          // binary frame (ArrayBuffer) -> create blob URL and set as src
          if (evt.data instanceof ArrayBuffer || evt.data instanceof Blob) {
            const arr = evt.data instanceof ArrayBuffer ? evt.data : evt.data.arrayBuffer()
            Promise.resolve(arr).then((buf) => {
              const blob = new Blob([buf], { type: 'image/jpeg' })
              const url = URL.createObjectURL(blob)
              if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl)
              lastBlobUrl = url
              const now = performance.now()
              if (lastFrameTimeRef.current !== null) {
                const deltaMs = now - lastFrameTimeRef.current
                if (deltaMs > 0) {
                  const instantFps = 1000 / deltaMs
                  setVideoFps(instantFps.toFixed(1))
                }
              }
              lastFrameTimeRef.current = now
              setVideoStreamSrc(url)
            }).catch((e) => console.error('blob create error', e))
            return
          }

          const data = JSON.parse(evt.data)
          pushDebug(`WS message: ${data.type}`)
          if (data.type === 'started') {
            setVideoStatus('Video received. Decoding and preparing frames...')
          } else if (data.type === 'started_preview') {
            setVideoFrames(0)
            setVideoFps(null)
            setVideoStatus('Video received. Showing placeholder preview...')
            setDetections([])
          } else if (data.type === 'preview') {
            setVideoFrames(data.frame + 1)
            setVideoFps(null)
            setVideoStatus('Showing initial preview frame...')
            setDetections([])
          } else if (data.type === 'frame_meta') {
            setVideoFrames(data.frame + 1)
            setVideoStatus(`Processing frame ${data.frame + 1}...`)
            const currentDetections = data.detections || []
            setDetections(currentDetections)
            if (currentDetections.length > 0) {
              setDetectHistory((prev) => {
                const nextItems = currentDetections
                  .map((d) => d.class_name)
                  .filter(Boolean)
                if (nextItems.length === 0) return prev
                const merged = [...prev, ...nextItems]
                return merged.slice(-80)
              })
            }
          } else if (data.type === 'done') {
            setVideoFrames(data.frames)
            setVideoStatus(`Done. Total frames: ${data.frames}`)
            pushDebug(`Stream done: frames=${data.frames}`)
            ws.close()
          } else if (data.type === 'error') {
            alert('Error: ' + data.message)
            setVideoStatus('Error: ' + data.message)
            pushDebug(`Server error: ${data.message}`)
            ws.close()
          }
        } catch (err) {
          console.error('ws message parse', err)
          pushDebug(`Message parse error: ${err?.message || err}`)
        }
      }

      ws.onerror = (e) => {
        console.error('ws error', e)
        alert('WebSocket error')
        setVideoStatus('WebSocket error')
        pushDebug('WebSocket error event')
        setLoading(false)
      }

      ws.onclose = () => {
        try {
          if (lastBlobUrl) {
            URL.revokeObjectURL(lastBlobUrl)
            lastBlobUrl = null
          }
        } catch (e) {
          console.warn('failed to revoke blob url', e)
        }
        pushDebug(`WebSocket closed. frames=${videoFrames ?? 'none'}, status=${videoStatus || 'empty'}`)
        if (videoFrames === null) {
          setVideoStatus('WebSocket closed before any frame arrived')
        }
        setLoading(false)
      }
    } catch (err) {
      console.error(err)
      alert('Error opening websocket')
      pushDebug(`Outer websocket error: ${err?.message || err}`)
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>VN Traffic Sign — Image Inference</h1>
      <div className="tabs">
        <button onClick={() => setTab('image')} className={tab==='image'?'active':''}>Image</button>
        <button onClick={() => setTab('video')} className={tab==='video'?'active':''}>Video (.mp4)</button>
      </div>

      {tab === 'image' && (
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
      )}

      {tab === 'video' && (
        <form onSubmit={handleVideoSubmit} className="form">
          <input
            type="file"
            accept="video/mp4"
            onChange={(e) => setFile(e.target.files[0])}
          />
          <button type="submit" disabled={!file || loading}>
            {loading ? 'Processing...' : 'Predict Video'}
          </button>
        </form>
      )}

      <div className="results">
        {annotated && tab === 'image' && (
          <div>
            <h2>Annotated</h2>
            <img src={annotated} alt="annotated" className="annotated" />
          </div>
        )}

        {tab === 'video' && (
          <div className="video-panel">
            <h2>Video Stream Preview</h2>
            <div style={{marginBottom: 8, color: '#555'}}>{videoStatus || 'Choose a video and click Predict Video'}</div>
            <div style={{marginBottom: 8, color: '#555'}}>
              FPS: {videoFps !== null ? videoFps : '...'}
            </div>
            <div className="video-layout">
              <div className="video-stack">
                <div ref={previewRef} className={"video-preview" + (isFs ? " fullwidth" : "")}> 
                {videoStreamSrc ? (
                  <>
                    <img src={videoStreamSrc} alt="stream" />
                    <button
                      className="fs-btn"
                      onClick={() => {
                        try {
                          if (!document.fullscreenElement) {
                            const el = previewRef.current
                            if (el.requestFullscreen) el.requestFullscreen()
                            else if (el.webkitRequestFullscreen) el.webkitRequestFullscreen()
                          } else {
                            if (document.exitFullscreen) document.exitFullscreen()
                            else if (document.webkitExitFullscreen) document.webkitExitFullscreen()
                          }
                        } catch (e) {
                          console.error('fullscreen error', e)
                        }
                      }}
                      title={isFs ? 'Exit fullscreen' : 'Fullscreen'}
                    >
                      {isFs ? '⤫' : '⤢'}
                    </button>
                  </>
                ) : (
                  <div style={{padding: 12, border: '1px dashed #aaa', color:'#999'}}>No stream yet</div>
                )}
              </div>
              </div>
              <div className="history-panel">
                <div className="history-header">
                  <h3>Detect History</h3>
                  <div className="history-meta">{videoFrames !== null ? `${videoFrames} frames` : 'No frames yet'}</div>
                </div>
                {(() => {
                  const grouped = new Map()
                  for (const name of [...detectHistory].reverse()) {
                    if (!name) continue
                    grouped.set(name, (grouped.get(name) || 0) + 1)
                  }
                  const historyItems = Array.from(grouped.entries())
                  return (
                <div className="history-list">
                  {historyItems.length === 0 ? (
                    <div className="history-empty">No detections yet</div>
                  ) : (
                    historyItems.map(([name]) => (
                      <div key={name} className="history-item">{VI_MAP[name] || name}</div>
                    ))
                  )}
                </div>
                  )
                })()}
                {videoAnnotated && <div style={{marginTop: 8}}>Final annotated video ready</div>}
              </div>
            </div>
          </div>
        )}

        {/* Detections shown in the right-side panel of the video view */}
      </div>
    </div>
  )
}
