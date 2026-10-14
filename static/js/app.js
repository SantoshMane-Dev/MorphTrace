document.addEventListener("DOMContentLoaded", () => {
	const form = document.querySelector(".upload-form");
	const fileInput = document.getElementById("image");
	const uploadBox = document.querySelector(".upload-box");
	const preview = document.querySelector(".upload-preview-image");
	const previewCaption = document.querySelector(".upload-preview-caption");
	const previewName = document.querySelector(".upload-preview-name");
	const uploadTitle = document.querySelector(".upload-title");
	const uploadMeta = document.querySelector(".upload-meta");
	const progressWrap = document.querySelector(".analyze-button-wrap");
	const progressBar = document.querySelector(".progress-bar");
	const progressTrack = document.querySelector(".progress-track");
	const progressText = document.querySelector(".progress-text");
	const submitButton = form?.querySelector("button[type='submit']");
	const replaceButton = document.querySelector(".rechoose-image");
	let previewUrl = null;

	const setProgress = (value) => {
		if (!progressBar || !progressTrack || !progressText) {
			return;
		}
		const progress = Math.max(0, Math.min(100, Math.round(value)));
		progressBar.style.width = `${progress}%`;
		progressTrack.setAttribute("aria-valuenow", String(progress));
		progressText.textContent = `${progress}%`;
	};

	const setSelectedFile = (file) => {
		if (!fileInput || !uploadBox || !preview || !previewCaption || !previewName || !uploadTitle || !uploadMeta) {
			return;
		}
		const extension = file?.name.split(".").pop()?.toLowerCase();
		if (!file || !["jpg", "jpeg", "png"].includes(extension)) {
			uploadBox.classList.remove("has-preview");
			preview.hidden = true;
			previewCaption.hidden = true;
			preview.removeAttribute("src");
			uploadTitle.textContent = "Drop a frame for review";
			uploadMeta.textContent = file ? "Choose a JPEG or PNG image" : "or choose one from your device";
			fileInput.value = "";
			return;
		}
		if (file.size > 16 * 1024 * 1024) {
			uploadMeta.textContent = "This image is over the 16 MB upload limit";
			fileInput.value = "";
			return;
		}
		if (previewUrl) {
			URL.revokeObjectURL(previewUrl);
		}
		previewUrl = URL.createObjectURL(file);
		preview.src = previewUrl;
		preview.hidden = false;
		previewCaption.hidden = false;
		previewName.textContent = file.name;
		uploadBox.classList.add("has-preview");
		uploadTitle.textContent = "Frame selected";
		uploadMeta.textContent = file.name;
	};

	const assignDroppedFiles = (input, files) => {
		if (!input || !files.length) {
			return;
		}
		const transfer = new DataTransfer();
		Array.from(files).forEach((file) => transfer.items.add(file));
		input.files = transfer.files;
		input.dispatchEvent(new Event("change", { bubbles: true }));
	};

	if (form && fileInput && uploadBox && preview && previewCaption && previewName && progressWrap && progressBar && progressTrack && progressText && submitButton) {
		fileInput.addEventListener("change", () => setSelectedFile(fileInput.files?.[0]));

		replaceButton?.addEventListener("click", (event) => {
			event.preventDefault();
			event.stopPropagation();
			fileInput.click();
		});

		["dragenter", "dragover"].forEach((eventName) => uploadBox.addEventListener(eventName, (event) => {
			event.preventDefault();
			uploadBox.classList.add("is-dragging");
		}));
		["dragleave", "drop"].forEach((eventName) => uploadBox.addEventListener(eventName, (event) => {
			event.preventDefault();
			uploadBox.classList.remove("is-dragging");
		}));
		uploadBox.addEventListener("drop", (event) => assignDroppedFiles(fileInput, event.dataTransfer?.files || []));

		form.addEventListener("submit", (event) => {
			event.preventDefault();
			if (!fileInput.files?.length || submitButton.disabled) {
				return;
			}

			const request = new XMLHttpRequest();
			const formData = new FormData(form);
			let analysisProgress = 0;
			let analysisTimer = null;

			progressWrap.classList.add("is-running", "is-uploading");
			progressWrap.classList.remove("is-analyzing", "is-complete", "is-error");
			progressText.classList.remove("is-error");
			progressText.textContent = "Uploading image... 0%";
			setProgress(0);
			submitButton.disabled = true;

			request.upload.addEventListener("progress", (progressEvent) => {
				if (!progressEvent.lengthComputable) {
					return;
				}
				const uploadPercent = (progressEvent.loaded / progressEvent.total) * 80;
				setProgress(uploadPercent);
				progressText.textContent = `Uploading image... ${Math.round(uploadPercent)}%`;
			});

			request.upload.addEventListener("load", () => {
				progressWrap.classList.remove("is-uploading");
				progressWrap.classList.add("is-analyzing");
				setProgress(80);
				progressText.textContent = "Image uploaded. Analyzing... 80%";
				analysisTimer = window.setInterval(() => {
					analysisProgress = Math.min(97, analysisProgress + 1);
					const value = 80 + analysisProgress * 0.17;
					setProgress(value);
					progressText.textContent = `Analyzing image... ${Math.round(value)}%`;
				}, 400);
			});

			request.addEventListener("load", () => {
				if (analysisTimer) {
					window.clearInterval(analysisTimer);
				}
				if (request.status >= 200 && request.status < 400 && request.responseURL) {
					progressWrap.classList.remove("is-analyzing");
					progressWrap.classList.add("is-complete");
					setProgress(100);
					progressText.textContent = "Analysis complete. 100%";
					window.location.assign(request.responseURL);
					return;
				}
				progressText.textContent = request.responseText || "Upload or analysis failed. Please try again.";
				progressText.classList.add("is-error");
				progressWrap.classList.add("is-error");
				progressWrap.classList.remove("is-running", "is-uploading", "is-analyzing");
				submitButton.disabled = false;
			});

			request.addEventListener("error", () => {
				if (analysisTimer) {
					window.clearInterval(analysisTimer);
				}
				progressText.textContent = "Could not connect. Please try again.";
				progressText.classList.add("is-error");
				progressWrap.classList.add("is-error");
				progressWrap.classList.remove("is-running", "is-uploading", "is-analyzing");
				submitButton.disabled = false;
			});

			request.open("POST", form.action);
			request.send(formData);
		});
	}

	const datasetFeedback = document.querySelector("[data-dataset-feedback]");
	document.querySelectorAll(".dataset-form").forEach((datasetForm) => {
		const input = datasetForm.querySelector("input[type='file']");
		const dropZone = datasetForm.querySelector(".dataset-drop");
		const selection = datasetForm.querySelector(".dataset-selection");
		const submit = datasetForm.querySelector("button[type='submit']");
		const card = datasetForm.closest("[data-class-card]");
		if (!input || !dropZone || !selection || !submit || !card) {
			return;
		}

		const showSelection = () => {
			const files = Array.from(input.files || []);
			selection.textContent = files.length
				? `${files.length} image${files.length === 1 ? "" : "s"} selected`
				: "Choose or drop a batch";
		};
		input.addEventListener("change", showSelection);
		["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
			event.preventDefault();
			dropZone.classList.add("is-dragging");
		}));
		["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
			event.preventDefault();
			dropZone.classList.remove("is-dragging");
		}));
		dropZone.addEventListener("drop", (event) => assignDroppedFiles(input, event.dataTransfer?.files || []));

		datasetForm.addEventListener("submit", async (event) => {
			event.preventDefault();
			if (!input.files?.length || submit.disabled) {
				return;
			}
			submit.disabled = true;
			submit.textContent = "Checking image files...";
			if (datasetFeedback) {
				datasetFeedback.classList.remove("is-error");
				datasetFeedback.textContent = "";
			}
			try {
				const response = await fetch(datasetForm.action, { method: "POST", body: new FormData(datasetForm) });
				const result = await response.json().catch(() => ({}));
				if (!response.ok) {
					throw new Error(result.message || "No valid images were added.");
				}
				card.querySelector("[data-image-count]").textContent = String(result.count);
				card.querySelector(".dataset-progress").value = result.count;
				card.querySelector(".dataset-target span:last-child").textContent = `${result.count} / 50`;
				input.value = "";
				showSelection();
				const rejected = result.errors?.length ? ` ${result.errors.length} file(s) skipped.` : "";
				if (datasetFeedback) {
					datasetFeedback.textContent = `${result.stored} image${result.stored === 1 ? "" : "s"} added to the ${card.dataset.classCard} bank.${rejected}`;
				}
			} catch (error) {
				if (datasetFeedback) {
					datasetFeedback.classList.add("is-error");
					datasetFeedback.textContent = error.message || "Could not add these images. Try again.";
				}
			} finally {
				submit.disabled = false;
				submit.innerHTML = `Add to ${card.dataset.classCard} bank <span aria-hidden="true">&#8594;</span>`;
			}
		});
	});
});
