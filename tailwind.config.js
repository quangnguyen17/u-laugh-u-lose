/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        palace: {
          purple: '#6D28D9',
          gold: '#F59E0B',
          ink: '#111827',
          fog: '#F9FAFB',
        },
      },
      boxShadow: { palace: '0 10px 25px rgba(109,40,217,0.25)' },
      borderRadius: { palace: '1.25rem' },
    },
  },
  plugins: [],
}
