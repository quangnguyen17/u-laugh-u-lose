import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as face from '@tensorflow-models/face-landmarks-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-converter'
import '@tensorflow/tfjs-backend-webgl'

export default function App() {
  const [screen, setScreen] = useState<'home' | 'active' | 'disqualified' | 'success'>('home')
  const [duration, setDuration] = useState<number>(5 * 60)
  const [remaining, setRemaining] = useState<number>(0)
  const [resultElapsed, setResultElapsed] = useState<number>(0)
  const [error, setError] = useState<string | null>(null)

  const poseBank = useMemo(
    () => [
      'Arms crossed, CEO stare',
      'Finger guns, deadpan',
      'Superhero stance, no smile',
      'T-Rex arms, serious face',
      'Teacup pinky out, stone-cold',
      'Power lunge, emotionless',
      'Cat claws up, stern look',
      'Phone-to-ear pretend call, poker face',
      'One eyebrow up (try), straight lips',
      'Hands on cheeks (Home Alone), zero grin',
      'Point at partner dramatically, stoic',
      'Pretend to sneeze mid-pose, no laugh',
      'Karate chop pose, calm',
      'Squat and point to camera, serious',
      'Wacky hat mime, unamused',
    ],
    []
  )

  const [poses, setPoses] = useState<string[]>([])
  const [poseIndex, setPoseIndex] = useState<number>(0)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)
  const modelRef = useRef<any>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const audioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const laughActiveRef = useRef<boolean>(false)

  const [config, setConfig] = useState({
    smileThreshold: 0.36,
    framesToTrigger: 6,
    audioEnabled: true,
    laughRmsThreshold: 0.06,
    laughHoldFrames: 6,
    flipHorizontal: true,
  })
  const triggerCountRef = useRef<number>(0)

  const timerRef = useRef<number | null>(null)
  const startedAtRef = useRef<number>(0)

  const sessionPoseCount = useMemo(() => {
    if (duration <= 5 * 60) return 3
    if (duration <= 15 * 60) return 6
    return 12
  }, [duration])

  function pickRandom<T>(arr: T[], count: number): T[] {
    const copy = [...arr]
    const out: T[] = []
    while (out.length < count && copy.length) {
      const i = Math.floor(Math.random() * copy.length)
      out.push(copy.splice(i, 1)[0])
    }
    return out
  }

  async function loadModel() {
    if (modelRef.current) return modelRef.current
    setError(null)
    try {
      await tf.setBackend('webgl')
      await tf.ready()
      const model = await face.createDetector(face.SupportedModels.MediaPipeFaceMesh, {
        runtime: 'tfjs',
        maxFaces: 1,
        refineLandmarks: true,
        detectorModelUrl: 'https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1',
        landmarkModelUrl:
          'https://tfhub.dev/mediapipe/tfjs-model/face_landmarks_detection/attention_mesh/1',
      })
      modelRef.current = model
      return model
    } catch (e: any) {
      setError('Failed to load ML model. Please refresh or try Chrome/Safari.')
      throw e
    }
  }

  async function startMedia() {
    setError(null)
    try {
      const constraints: MediaStreamConstraints = {
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: config.audioEnabled,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await (videoRef.current as HTMLVideoElement).play()
      }
      if (config.audioEnabled) setupAudio(stream)
    } catch (e: any) {
      setError('Camera/mic permission denied or unavailable. Ensure HTTPS and allow access.')
      console.error(e)
      throw e
    }
  }

  function setupAudio(stream: MediaStream) {
    cleanupAudio()
    try {
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)()
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 2048
      const source = ctx.createMediaStreamSource(stream)
      source.connect(analyser)
      audioCtxRef.current = ctx
      analyserRef.current = analyser
      audioSourceRef.current = source
      laughActiveRef.current = true
    } catch (e) {
      console.warn('Audio init failed', e)
      laughActiveRef.current = false
    }
  }

  function cleanupAudio() {
    try {
      analyserRef.current?.disconnect()
      audioSourceRef.current?.disconnect()
      audioCtxRef.current?.close()
    } catch {}
    analyserRef.current = null
    audioSourceRef.current = null
    audioCtxRef.current = null
    laughActiveRef.current = false
  }

  function cleanupMedia() {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    cleanupAudio()
  }

  function resetAll() {
    setScreen('home')
    setRemaining(0)
    setResultElapsed(0)
    setPoseIndex(0)
    setPoses([])
    triggerCountRef.current = 0
    if (timerRef.current) window.clearInterval(timerRef.current)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    cleanupMedia()
  }

  useEffect(() => () => resetAll(), [])

  function formatTime(s: number) {
    const m = Math.floor(s / 60)
      .toString()
      .padStart(2, '0')
    const ss = Math.floor(s % 60)
      .toString()
      .padStart(2, '0')
    return `${m}:${ss}`
  }

  function mouthMetrics(annotations: any) {
    const ML = annotations.mouthLeft?.[0]
    const MR = annotations.mouthRight?.[0]
    const U = annotations.lipsUpperOuter?.[5] || annotations.lipsUpperInner?.[5]
    const L = annotations.lipsLowerOuter?.[5] || annotations.lipsLowerInner?.[5]
    const leftCheek = annotations.leftCheek?.[0] || annotations.silhouette?.[234]
    const rightCheek = annotations.rightCheek?.[0] || annotations.silhouette?.[454]

    if (!ML || !MR || !U || !L || !leftCheek || !rightCheek) return null

    const dx = MR[0] - ML[0]
    const dy = MR[1] - ML[1]
    const mouthWidth = Math.hypot(dx, dy)
    const mouthHeight = Math.hypot(U[0] - L[0], U[1] - L[1])
    const faceWidth = Math.hypot(rightCheek[0] - leftCheek[0], rightCheek[1] - leftCheek[1])

    const widthNorm = mouthWidth / faceWidth
    const heightNorm = mouthHeight / faceWidth
    const smileScore = widthNorm - 0.6 * heightNorm
    return { widthNorm, heightNorm, smileScore }
  }

  function detectLaughRms(): boolean {
    if (!laughActiveRef.current || !analyserRef.current) return false
    const analyser = analyserRef.current
    const buf = new Uint8Array(analyser.fftSize)
    analyser.getByteTimeDomainData(buf)
    let sum = 0
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128
      sum += v * v
    }
    const rms = Math.sqrt(sum / buf.length)
    return rms > config.laughRmsThreshold
  }

  async function startChallenge() {
    try {
      const selectedPoses = pickRandom(poseBank, sessionPoseCount)
      setPoses(selectedPoses)
      setPoseIndex(0)

      await startMedia()
      await loadModel()

      setRemaining(duration)
      startedAtRef.current = Date.now()
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = window.setInterval(() => {
        const elapsed = Math.floor((Date.now() - startedAtRef.current) / 1000)
        const left = Math.max(0, duration - elapsed)
        setRemaining(left)
        if (left <= 0) endChallenge(true)
      }, 1000)

      setScreen('active')
      triggerCountRef.current = 0
      runDetectionLoop()
    } catch (e) {
      console.error(e)
    }
  }

  function endChallenge(success: boolean) {
    if (timerRef.current) window.clearInterval(timerRef.current)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    const elapsed = Math.floor((Date.now() - startedAtRef.current) / 1000)
    setResultElapsed(elapsed)
    cleanupMedia()
    setScreen(success ? 'success' : 'disqualified')
  }

  async function runDetectionLoop() {
    const model = modelRef.current
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!model || !video || !canvas) return

    const ctx = canvas.getContext('2d')
    const loop = async () => {
      if (!videoRef.current || !ctx) return
      canvas.width = video.videoWidth || 1280
      canvas.height = video.videoHeight || 720

      ctx.save()
      if (config.flipHorizontal) {
        ctx.translate(canvas.width, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      ctx.restore()

      let smiling = false
      try {
        const faces = await model.estimateFaces(video, {
          flipHorizontal: config.flipHorizontal,
          staticImageMode: false,
        })
        if (faces && faces[0]) {
          const f = faces[0]
          const metrics = mouthMetrics(f.annotations)
          if (metrics) {
            const { smileScore } = metrics
            smiling = smileScore > config.smileThreshold
            ctx.fillStyle = 'rgba(109,40,217,0.35)'
            ctx.fillRect(8, 8, 240, 54)
            ctx.fillStyle = 'white'
            ctx.font = '14px ui-sans-serif, system-ui'
            ctx.fillText(`smileScore: ${smileScore.toFixed(3)}`, 16, 28)
            ctx.fillText(`threshold: ${config.smileThreshold.toFixed(2)}`, 16, 48)
          }
        }
      } catch (e) {}

      const laughed = config.audioEnabled && detectLaughRms()
      const violated = smiling || laughed

      if (violated) triggerCountRef.current += 1
      else triggerCountRef.current = 0

      if (ctx) {
        ctx.strokeStyle = violated ? '#EF4444' : '#10B981'
        ctx.lineWidth = 8
        ctx.strokeRect(20, 20, canvas.width - 40, canvas.height - 40)
      }

      if (triggerCountRef.current >= config.framesToTrigger) {
        endChallenge(false)
        return
      }

      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
  }

  function nextPose() {
    setPoseIndex((p) => Math.min(poses.length - 1, p + 1))
  }
  function onTryAgain() {
    resetAll()
  }

  const Header = () => (
    <div className="w-full flex items-center justify-between mb-4">
      <h1 className="text-2xl md:text-3xl font-extrabold text-palace.purple">
        Pose Palace — U Laugh, U Lose
      </h1>
      <span className="text-xs md:text-sm text-palace.gold">Smile Detector Challenge</span>
    </div>
  )

  const Card: React.FC<{ children: React.ReactNode; className?: string }> = ({
    children,
    className,
  }) => (
    <div
      className={`rounded-2xl shadow-lg p-6 bg-white/90 backdrop-blur ring-1 ring-palace.purple/10 ${
        className || ''
      }`}
    >
      {children}
    </div>
  )

  const HomeScreen = () => (
    <div className="max-w-3xl mx-auto p-4">
      <Header />
      <Card>
        <p className="mb-4 text-gray-700">
          Team up and keep a straight face while you complete random poses. If the app detects a
          smile or laughter, you’re disqualified.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          {[5, 15, 30].map((m) => (
            <button
              key={m}
              onClick={() => setDuration(m * 60)}
              className={`rounded-xl border px-4 py-3 text-left hover:shadow transition ${
                duration === m * 60 ? 'bg-black text-white' : 'border-gray-300'
              }`}
            >
              <div className="text-lg font-semibold">{m} minutes</div>
              <div className="text-xs text-gray-500">
                {m === 5 ? '3 poses' : m === 15 ? '6 poses' : '12 poses'}
              </div>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-3 mb-4">
          <label className="text-sm">Audio laugh detection</label>
          <input
            type="checkbox"
            checked={config.audioEnabled}
            onChange={(e) => setConfig({ ...config, audioEnabled: e.target.checked })}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
          <div>
            <label className="text-xs text-gray-500">Smile threshold</label>
            <input
              type="range"
              min={0.2}
              max={0.6}
              step={0.01}
              value={config.smileThreshold}
              onChange={(e) =>
                setConfig({
                  ...config,
                  smileThreshold: parseFloat((e.target as HTMLInputElement).value),
                })
              }
              className="w-full"
            />
            <div className="text-xs">{config.smileThreshold.toFixed(2)}</div>
          </div>
          <div>
            <label className="text-xs text-gray-500">Frames to trigger</label>
            <input
              type="number"
              min={3}
              max={20}
              value={config.framesToTrigger}
              onChange={(e) =>
                setConfig({
                  ...config,
                  framesToTrigger: parseInt((e.target as HTMLInputElement).value || '6'),
                })
              }
              className="w-full border rounded px-2 py-1"
            />
          </div>
          <div>
            <label className="text-xs text-gray-500">Laugh RMS threshold</label>
            <input
              type="number"
              step={0.01}
              min={0.02}
              max={0.2}
              value={config.laughRmsThreshold}
              onChange={(e) =>
                setConfig({
                  ...config,
                  laughRmsThreshold: parseFloat((e.target as HTMLInputElement).value || '0.06'),
                })
              }
              className="w-full border rounded px-2 py-1"
            />
          </div>
        </div>

        <button
          onClick={startChallenge}
          className="w-full md:w-auto rounded-2xl px-6 py-3 bg-black text-white font-semibold shadow-palace hover:opacity-95"
        >
          Start Challenge
        </button>

        {error && <p className="mt-4 text-red-600 text-sm">{error}</p>}

        <div className="mt-6 text-xs text-gray-500">
          Tip: Run over HTTPS. On iPhone/iPad Safari, allow camera/mic and keep the screen awake.
        </div>
      </Card>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <div className="font-semibold mb-2 text-palace.purple">Rules</div>
          <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
            <li>No laughing out loud.</li>
            <li>No smiling or laughing at the camera.</li>
            <li>No visible laughs in any captured photo.</li>
            <li>AI detector monitors face and optional microphone.</li>
            <li>Need to laugh? Turn fully away from the camera.</li>
          </ul>
        </Card>
        <Card>
          <div className="font-semibold mb-2 text-palace.purple">Prizes</div>
          <p className="text-sm text-gray-700">Win: 50% off any session of your choice.</p>
        </Card>
        <Card>
          <div className="font-semibold mb-2 text-palace.purple">Session → Pose Count</div>
          <ul className="text-sm text-gray-700 space-y-1">
            <li>5 minutes → 3 random poses</li>
            <li>15 minutes → 6 random poses</li>
            <li>30 minutes → 12 random poses</li>
          </ul>
        </Card>
      </div>
    </div>
  )

  const ActiveScreen = () => (
    <div className="max-w-5xl mx-auto p-4">
      <Header />
      <div className="grid md:grid-cols-[2fr_1fr] gap-4">
        <Card>
          <div className="relative w-full">
            <video ref={videoRef} playsInline muted className="hidden" />
            <canvas
              ref={canvasRef}
              className="w-full rounded-2xl shadow ring-1 ring-palace.purple/10"
            />
          </div>
          <div className="mt-3 flex items-center justify-between">
            <div className="text-lg font-semibold tabular-nums">⏳ {formatTime(remaining)}</div>
            <button
              className="rounded-xl border px-4 py-2 text-sm hover:bg-gray-50"
              onClick={() => endChallenge(false)}
            >
              Give Up
            </button>
          </div>
        </Card>
        <div className="space-y-4">
          <Card>
            <div className="text-sm text-gray-500">Current Pose</div>
            <div className="text-lg font-semibold">{poses[poseIndex] || 'Hold steady'}</div>
            <div className="text-xs text-gray-500 mt-2">
              Pose {poseIndex + 1} of {poses.length}
            </div>
            <button
              onClick={() => setPoseIndex((p) => Math.min(poses.length - 1, p + 1))}
              className="mt-3 rounded-xl border px-3 py-2 text-sm hover:bg-gray-50"
            >
              Next Pose
            </button>
          </Card>
          <Card>
            <div className="font-semibold mb-2 text-palace.purple">Tips</div>
            <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
              <li>Keep lips relaxed and corners down.</li>
              <li>Think of something boring. Seriously.</li>
              <li>If you must break, turn fully away first.</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )

  const DisqualifiedScreen = () => (
    <div className="max-w-xl mx-auto p-4 text-center">
      <Header />
      <Card>
        <div className="text-3xl font-extrabold text-red-600">❌ Disqualified</div>
        <div className="mt-2 text-gray-700">
          A smile or laugh was detected! You lasted {formatTime(resultElapsed)}.
        </div>
        <div className="mt-6 flex gap-3 justify-center">
          <button
            onClick={() => resetAll()}
            className="rounded-xl border px-5 py-3 font-semibold hover:bg-gray-50"
          >
            Try Again
          </button>
        </div>
      </Card>
    </div>
  )

  const SuccessScreen = () => (
    <div className="max-w-xl mx-auto p-4 text-center">
      <Header />
      <Card>
        <div className="text-3xl font-extrabold text-green-600">✅ You did it!</div>
        <div className="mt-2 text-gray-700">
          You completed the {Math.round(duration / 60)}-minute challenge without smiling.
        </div>
        <div className="mt-6 flex gap-3 justify-center">
          <button
            onClick={() => resetAll()}
            className="rounded-xl border px-5 py-3 font-semibold hover:bg-gray-50"
          >
            Play Again
          </button>
        </div>
      </Card>
    </div>
  )

  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-palace.fog to-gray-100 text-gray-900">
      {screen === 'home' && <HomeScreen />}
      {screen === 'active' && <ActiveScreen />}
      {screen === 'disqualified' && <DisqualifiedScreen />}
      {screen === 'success' && <SuccessScreen />}
    </div>
  )
}
