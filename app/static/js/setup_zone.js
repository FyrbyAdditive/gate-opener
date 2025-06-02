document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearZoneBtn = document.getElementById('clearZoneBtn');
    const undoPointBtn = document.getElementById('undoPointBtn');
    const saveZoneBtn = document.getElementById('saveZoneBtn');
    const currentPointsDisplay = document.getElementById('currentPointsDisplay');

    let points = []; // Stores {x, y} normalized coordinates

    // Set canvas dimensions using the global constants from setup.html
    function resizeCanvas() {
        // videoWidth and videoHeight are global JS constants from the setup.html template
        canvas.width = videoWidth;
        canvas.height = videoHeight;
        draw(); // Redraw with new dimensions
    }

    // Initialize canvas size and load current zone
    resizeCanvas(); // Call immediately as videoWidth/Height are already defined
    window.addEventListener('resize', resizeCanvas); // Adjust canvas on window resize (optional, if container might change)

    // Tell backend to HIDE the persistent zone from the video stream for this page
    fetch('/set_stream_zone_visibility', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({visible: false})
    }).then(response => response.json())
      .then(data => console.log('Setup page: stream zone visibility set to false.', data));

    loadCurrentZone();

    canvas.addEventListener('click', function (event) {
        const rect = canvas.getBoundingClientRect(); // Gets the actual size and position of canvas on screen

        // Prevent division by zero if canvas is not visible or has no dimensions
        if (rect.width === 0 || rect.height === 0) {
            console.warn("Canvas display dimensions are zero. Click ignored.");
            return;
        }

        // Calculate click relative to the canvas element
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        // Normalize points based on the canvas's actual displayed size (rect.width, rect.height).
        // This ensures the normalized coordinates (0-1) correctly represent the click's
        // proportional position on the video, regardless of how the video (and canvas) is scaled for display.
        points.push({ x: x / rect.width, y: y / rect.height });
        draw();
        updateSaveButtonState();
        updatePointsDisplay();
    });

    clearZoneBtn.addEventListener('click', function () {
        points = [];
        draw();
        updateSaveButtonState();
        updatePointsDisplay();
    });

    undoPointBtn.addEventListener('click', function () {
        if (points.length > 0) {
            points.pop();
            draw();
            updateSaveButtonState();
            updatePointsDisplay();
        }
    });

    saveZoneBtn.addEventListener('click', function () {
        const pointsStr = points.map(p => `${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(';');
        
        fetch('/setup/save_zone', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                // Add CSRF token header if you implement CSRF protection
            },
            body: JSON.stringify({ points_str: pointsStr }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Use Bootstrap's alert or a more subtle notification if available
                alert('Activation zone saved successfully!'); // Simple alert
                // Consider using flash messages via a redirect or dynamically adding to page
            } else {
                alert('Error saving zone: ' + data.message);
            }
            loadCurrentZone(); // Reload to show saved state or errors
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('An error occurred while saving the zone.');
        });
    });

    function loadCurrentZone() {
        fetch('/setup/get_zone_points')
            .then(response => response.json())
            .then(data => {
                points = [];
                if (data.zone_points_str) {
                    const pairs = data.zone_points_str.split(';');
                    pairs.forEach(pair => {
                        const coords = pair.split(',');
                        if (coords.length === 2) {
                            points.push({ x: parseFloat(coords[0]), y: parseFloat(coords[1]) });
                        }
                    });
                }
                draw();
                updateSaveButtonState();
                updatePointsDisplay();
            });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas

        if (points.length === 0) return;

        // --- 1. Draw the main polygon lines and fill (if applicable) ---
        ctx.beginPath(); // Start path for the polygon
        ctx.strokeStyle = 'rgba(255, 0, 255, 0.8)'; // Magenta for drawing lines
        ctx.lineWidth = 2;

        points.forEach((p, index) => {
            const absX = p.x * canvas.width;
            const absY = p.y * canvas.height;
            if (index === 0) {
                ctx.moveTo(absX, absY);
            } else {
                ctx.lineTo(absX, absY);
            }
        });

        // Stroke the lines connecting the points if there are at least two points
        if (points.length >= 2) {
            ctx.stroke();
        }

        // If there are enough points, close the path, fill it, and re-stroke the closed shape
        if (points.length >= 3) {
            ctx.closePath(); // Connects the last point to the first
            ctx.fillStyle = 'rgba(255, 0, 255, 0.15)'; // Light fill for the zone
            ctx.fill();
            ctx.stroke(); // Re-stroke to ensure the closing line is also drawn with the same style
        }

        // --- 2. Draw the small circles for each point on top ---
        points.forEach((p) => {
            const absX = p.x * canvas.width;
            const absY = p.y * canvas.height;
            
            ctx.fillStyle = 'rgba(255, 0, 255, 0.6)';
            ctx.strokeStyle = 'rgba(255, 0, 255, 0.8)'; // Border for the points
            ctx.lineWidth = 1; // Can be different from polygon line width
            ctx.beginPath(); // New path for each circle
            ctx.arc(absX, absY, 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });
    }

    function updateSaveButtonState() {
        // Enable save if there are at least 3 points, or if points array is empty (to clear the zone)
        saveZoneBtn.disabled = !(points.length >= 3 || points.length === 0);
    }

    function updatePointsDisplay() {
        if (points.length === 0) {
            currentPointsDisplay.textContent = "No points defined. Click to add.";
        } else {
            currentPointsDisplay.textContent = points.map(p => `(${p.x.toFixed(3)}, ${p.y.toFixed(3)})`).join('; ');
        }
    }
});