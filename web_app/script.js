// Custom cursor movement
const cursor = document.querySelector('.cursor-circle');
document.addEventListener('mousemove', (e) => {
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
});
document.addEventListener('mouseleave', () => cursor.style.opacity = '0');
document.addEventListener('mouseenter', () => cursor.style.opacity = '1');

// ========== Manual data for Base Game ==========
const baseData = {
    peak: 612,
    avg: 350,
    consistency: 34,
    chartData: [12, 18, 25, 32, 40, 45, 52, 58, 63, 70, 75, 80, 83, 87, 86, 85, 84, 87]
};

// ========== Auto-load CSV files ==========
const models = [
    { id: 'dqn', file: 'training_log.csv', scoreCol: 'score', fallbackScores: [100, 150, 200, 250] },
    { id: 'hybrid', file: 'hybrid_agent_log.csv', scoreCol: 'hybrid_score', fallbackScores: [50, 100, 150] },
    { id: 'rule', file: 'rule_bot_log.csv', scoreCol: 'score', fallbackScores: [] }
];

// Helper: compute metrics from array of scores
function computeStats(scores) {
    const peak = Math.max(...scores);
    const avg = scores.reduce((a,b) => a+b, 0) / scores.length;
    const std = Math.sqrt(scores.map(x => (x - avg) ** 2).reduce((a,b) => a+b, 0) / scores.length);
    const cv = std / avg;
    const consistency = Math.max(0, Math.min(100, 100 - cv * 100));
    return { peak, avg, consistency };
}

// Update DOM and chart for a model
function updateModelUI(modelId, stats, chartData, errorMsg = null) {
    const peakEl = document.getElementById(`${modelId}-peak`);
    const avgEl = document.getElementById(`${modelId}-avg`);
    const consEl = document.getElementById(`${modelId}-cons`);
    const noteEl = document.querySelector(`.stat-card[data-model="${modelId}"] .stat-note`);
    
    if (peakEl) peakEl.innerText = Math.round(stats.peak);
    if (avgEl) avgEl.innerText = Math.round(stats.avg);
    if (consEl) consEl.innerText = `${Math.round(stats.consistency)}%`;
    if (noteEl && errorMsg) noteEl.innerHTML = `⚠️ ${errorMsg}`;
    else if (noteEl && !errorMsg) noteEl.innerHTML = `📊 Loaded ${chartData.length} episodes`;

    const canvas = document.getElementById(`${modelId}-chart`);
    if (!canvas) return;
    if (canvas.chart) canvas.chart.destroy();
    const ctx = canvas.getContext('2d');
    const color = modelId === 'dqn' ? '#8b5cf6' : (modelId === 'hybrid' ? '#10b981' : '#f59e0b');
    canvas.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: chartData.map((_, i) => i+1),
            datasets: [{
                data: chartData,
                borderColor: color,
                borderWidth: 2,
                fill: false,
                pointRadius: 1,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `Score: ${ctx.raw}` } } },
            scales: { x: { display: false }, y: { display: false } }
        }
    });
}

// Load a CSV file and update model
async function loadModelCSV(model) {
    try {
        const response = await fetch(model.file);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const text = await response.text();
        const rows = text.trim().split('\n');
        if (rows.length < 2) throw new Error('Empty file');
        const headers = rows[0].split(',');
        let scoreIdx = headers.findIndex(h => h.toLowerCase().trim() === model.scoreCol.toLowerCase().trim());
        if (scoreIdx === -1) scoreIdx = headers.findIndex(h => h === 'score');
        if (scoreIdx === -1) throw new Error(`Column "${model.scoreCol}" not found`);

        const scores = [];
        for (let i = 1; i < rows.length; i++) {
            const cols = rows[i].split(',');
            const val = parseFloat(cols[scoreIdx]);
            if (!isNaN(val)) scores.push(val);
        }
        if (scores.length === 0) throw new Error('No numeric scores');

        const stats = computeStats(scores);
        let chartData = scores;
        if (scores.length > 200) chartData = scores.filter((_, i) => i % Math.ceil(scores.length/200) === 0);
        updateModelUI(model.id, stats, chartData);
    } catch (err) {
        console.warn(`Failed to load ${model.file}:`, err.message);
        // Use fallback data
        const fallbackScores = model.fallbackScores || [0];
        const stats = computeStats(fallbackScores);
        updateModelUI(model.id, stats, fallbackScores, `Could not load ${model.file}`);
    }
}

// Initialize Base Game (manual)
function initBaseGame() {
    updateModelUI('base', 
        { peak: baseData.peak, avg: baseData.avg, consistency: baseData.consistency }, 
        baseData.chartData,
        null
    );
}

// Run everything
initBaseGame();
models.forEach(m => loadModelCSV(m));

// Optional hover effect for stat cards
document.querySelectorAll('.stat-card').forEach(card => {
    card.addEventListener('mouseenter', () => cursor.style.transform = 'translate(-50%, -50%) scale(1.2)');
    card.addEventListener('mouseleave', () => cursor.style.transform = 'translate(-50%, -50%) scale(1)');
});