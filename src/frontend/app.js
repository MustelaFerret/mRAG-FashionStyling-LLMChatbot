const { useEffect, useMemo, useRef, useState } = React;

const INTENT_LABELS = {
    similar_items: "Similar Picks",
    color_variant: "Color Variants",
    graph_pairing: "Outfit Pairing",
    chit_chat: "Chit Chat",
    composite_intent: "Intent Confirmation",
};

const INTENT_OPTIONS = [
    {
        id: "similar_items",
        label: "Similar Items",
        description: "Find items similar to the reference or description.",
    },
    {
        id: "graph_pairing",
        label: "Outfit Pairing",
        description: "Find complementary pieces to wear with the reference item.",
    },
    {
        id: "color_variant",
        label: "Color Variant",
        description: "Find the same item in a different color.",
    },
];

const DEFAULT_BOOTSTRAP = {
    app: {
        name: "Atelier mRAG",
        tagline: "AI Fashion Styling Studio",
        api_version: "2",
        theme: "retro-vintage",
    },
    suggested_prompts: [
        "Find pants that match this top",
        "Show color alternatives for #article_id",
        "Build a smart-casual outfit from this item",
        "Find a cleaner minimal version of this look",
    ],
};

function uid() {
    if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    return "id-" + Date.now() + "-" + Math.random().toString(16).slice(2);
}

function getSessionId() {
    const key = "mrag_session_id";
    let sid = localStorage.getItem(key);
    if (!sid) {
        sid = uid();
        localStorage.setItem(key, sid);
    }
    return sid;
}

function formatArticleId(value) {
    const str = String(value || "").trim();
    return /^\d+$/.test(str) ? str.padStart(10, "0") : str;
}

function imagePathFromId(articleId) {
    const id = formatArticleId(articleId);
    if (!id) return "";
    return "/images/" + id.substring(0, 3) + "/" + id + ".jpg";
}

function normalizeItems(items) {
    return (items || [])
        .map((item) => ({
            ...item,
            article_id: formatArticleId(item.article_id),
            image_url: item.image_url || item.image_path || imagePathFromId(item.article_id),
            title: item.title || item.name || item.product_type || "Item",
            subtitle: item.subtitle || item.colour_group || "",
        }))
        .filter((item) => item.article_id);
}

function ProductModal({ items, index, onClose, onNavigate }) {
    if (!Array.isArray(items) || items.length === 0) return null;
    const safeIndex = Math.max(0, Math.min(index || 0, items.length - 1));
    const item = items[safeIndex];
    if (!item) return null;

    const aid = formatArticleId(item.article_id);
    const title = item.title || item.product_type || "Product";
    const subtitle = item.subtitle || item.colour_group || "-";
    const hasNav = items.length > 1;
    const prevIndex = (safeIndex - 1 + items.length) % items.length;
    const nextIndex = (safeIndex + 1) % items.length;

    useEffect(() => {
        function onKeyDown(event) {
            if (event.key === "Escape") {
                onClose();
                return;
            }
            if (!hasNav) return;
            if (event.key === "ArrowLeft") {
                onNavigate(prevIndex);
            } else if (event.key === "ArrowRight") {
                onNavigate(nextIndex);
            }
        }

        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [hasNav, nextIndex, onClose, onNavigate, prevIndex]);

    return (
        <div id="product-modal" onClick={onClose}>
            {hasNav && (
                <button
                    className="product-modal-nav product-modal-prev"
                    onClick={(e) => {
                        e.stopPropagation();
                        onNavigate(prevIndex);
                    }}
                    aria-label="Previous product"
                >
                    <span aria-hidden="true">&lt;</span>
                </button>
            )}
            {hasNav && (
                <button
                    className="product-modal-nav product-modal-next"
                    onClick={(e) => {
                        e.stopPropagation();
                        onNavigate(nextIndex);
                    }}
                    aria-label="Next product"
                >
                    <span aria-hidden="true">&gt;</span>
                </button>
            )}
            <div className="product-modal-panel relative" onClick={(e) => e.stopPropagation()}>
                <button className="absolute top-4 right-5 text-3xl leading-none t-muted hover:t-ink z-10" onClick={onClose} aria-label="Close">&times;</button>
                <div className="product-modal-media">
                    <img className="product-modal-image" src={item.image_url || imagePathFromId(aid)} alt={title} />
                </div>
                <div className="product-modal-details">
                    <p className="kicker mb-4">The Piece</p>
                    <h3 className="font-display text-5xl leading-[0.95] mb-2">{title}</h3>
                    <p className="font-code text-xs t-faint">#{aid}</p>
                    <div className="spec-grid">
                        <div><span className="spec-label">Color</span><span className="spec-value">{subtitle || "—"}</span></div>
                        <div><span className="spec-label">Fit</span><span className="spec-value">{item.fit || "—"}</span></div>
                        <div><span className="spec-label">Occasion</span><span className="spec-value">{item.occasion || "—"}</span></div>
                        <div><span className="spec-label">Season</span><span className="spec-value">{item.seasonality || "—"}</span></div>
                    </div>
                    <p className="kicker mt-6 mb-3">Notes</p>
                    <p className="text-sm leading-relaxed t-soft">{item.description || "No description available."}</p>
                </div>
            </div>
        </div>
    );
}

function HeroScreen({ bootstrap, onStart }) {
    const appName = (bootstrap && bootstrap.app && bootstrap.app.name) || DEFAULT_BOOTSTRAP.app.name;
    const tagline = (bootstrap && bootstrap.app && bootstrap.app.tagline) || DEFAULT_BOOTSTRAP.app.tagline;
    const rootRef = useRef(null);

    useEffect(() => {
        if (!window.gsap || !rootRef.current) return undefined;

        const q = window.gsap.utils.selector(rootRef);
        const tl = window.gsap.timeline({ defaults: { ease: "power3.out" } });

        tl.from(q(".hero-badge"), { y: 14, opacity: 0, duration: 0.45 })
            .from(q(".hero-title"), { y: 24, opacity: 0, duration: 0.65 }, "-=0.16")
            .from(q(".hero-sub"), { y: 16, opacity: 0, duration: 0.52 }, "-=0.28")
            .from(q(".hero-enter-btn"), { y: 8, opacity: 0, duration: 0.36 }, "-=0.12");

        return () => tl.kill();
    }, []);

    return (
        <section className="hero" ref={rootRef}>
            <div className="max-w-3xl">
                <p className="hero-badge mb-6">{appName} — {tagline}</p>
                <h1 className="hero-title">Find your<br /><em>silhouette</em>.</h1>
                <p className="hero-sub">
                    Upload a reference, refine the intent, and let the atelier compose looks tuned to your taste.
                </p>
                <button onClick={onStart} className="btn-primary hero-enter-btn mt-10 px-12 py-4">
                    Enter the atelier
                </button>
            </div>
            <HeroTicker />
        </section>
    );
}

function HeroTicker() {
    const words = ["Tailoring", "Colour", "Texture", "Silhouette", "Drape", "Proportion", "Palette", "Layering"];
    const group = (
        <span className="hero-ticker-group">
            {words.map((w, i) => (
                <React.Fragment key={w + i}>
                    <span>{w}</span>
                    <span className="dot">&bull;</span>
                </React.Fragment>
            ))}
        </span>
    );
    return (
        <div className="hero-ticker" aria-hidden="true">
            <div className="hero-ticker-track">
                {group}
                {group}
            </div>
        </div>
    );
}

function StatusRibbon({ meta, sessionId }) {
    return (
        <div className="status-ribbon">
            <span className="masthead-mark font-display">Atelier<span className="t-accent">·</span>mRAG</span>
            <span className="status-pill" style={{ marginLeft: "auto" }}>Session {sessionId.slice(0, 8)}</span>
            {meta && <span className="status-pill">Results {meta.result_count || 0}</span>}
            <span className="status-pill">{meta ? "Live" : "Ready"}</span>
        </div>
    );
}

function UserMessage({ message }) {
    return (
        <div className="chat-message flex flex-col items-end gap-2">
            {message.image && <img src={message.image} className="max-w-[220px] md:max-w-[260px] rounded border border-black/10" alt="Uploaded" />}
            {message.text && <div className="msg-user">{message.text}</div>}
        </div>
    );
}

function ItemCard({ item, onOpen }) {
    const aid = formatArticleId(item.article_id);
    const title = item.title || item.product_type || "Item";
    const subtitle = item.subtitle || item.colour_group || "";

    return (
        <div
            className={"item-card " + (item.is_anchor ? "anchor" : "")}
            onClick={onOpen}
        >
            <div className="card-frame">
                <img src={item.image_url || imagePathFromId(aid)} alt={title} />
            </div>
            <p className="card-name">{title}</p>
            <div className="card-meta">
                <span className="card-color">{subtitle || "—"}</span>
                {item.is_anchor
                    ? <span className="card-focus-tag">In focus</span>
                    : <span className="card-id">{aid}</span>}
            </div>
        </div>
    );
}

function IntentPicker({ options, onSelect, disabled }) {
    const optionSet = new Set(Array.isArray(options) ? options : []);
    const available = INTENT_OPTIONS.filter((opt) => optionSet.has(opt.id));
    if (available.length === 0) return null;

    return (
        <div className="intent-picker">
            <p className="intent-picker-title">Confirm intent</p>
            <div className="intent-picker-grid">
                {available.map((opt) => (
                    <button
                        key={opt.id}
                        className="intent-option"
                        disabled={disabled}
                        onClick={() => onSelect(opt.id)}
                    >
                        <span className="intent-option-title">{opt.label}</span>
                        <span className="intent-option-desc">{opt.description}</span>
                    </button>
                ))}
            </div>
        </div>
    );
}

function AiMessage({ message, onOpenItem, onConfirmIntent }) {
    const intentLabel = (message.intentInfo && message.intentInfo.label) || INTENT_LABELS[message.intent] || "";
    const intentDescription = (message.intentInfo && message.intentInfo.description) || "";
    const cards = Array.isArray(message.cards) && message.cards.length > 0 ? message.cards : (message.items || []);
    const intentOptions = Array.isArray(message.intentOptions) ? message.intentOptions : [];
    const showIntentPicker = message.intent === "composite_intent" && intentOptions.length > 0 && !message.intentResolved;

    return (
        <div className="chat-message flex flex-col items-start gap-3">
            <div className={"msg-ai " + (message.error ? "t-accent" : "")}>{message.text}</div>
            {intentLabel && <span className="intent-chip">{intentLabel}</span>}
            {intentDescription && <p className="text-xs t-muted -mt-1">{intentDescription}</p>}

            {showIntentPicker && (
                <IntentPicker
                    options={intentOptions}
                    disabled={message.intentResolved}
                    onSelect={(intentId) => onConfirmIntent && onConfirmIntent(message, intentId)}
                />
            )}

            {Array.isArray(cards) && cards.length > 0 && (
                <div className="cards-row w-full">
                    {cards.map((item, index) => (
                        <ItemCard key={String(item.article_id) + "-" + index} item={item} onOpen={() => onOpenItem(item, cards)} />
                    ))}
                </div>
            )}
        </div>
    );
}

function QuickActions({ actions, onPick }) {
    if (!Array.isArray(actions) || actions.length === 0) return null;

    return (
        <div className="quick-actions">
            {actions.map((action, index) => (
                <button key={action + "-" + index} className="quick-action-chip" onClick={() => onPick(action)}>
                    {action}
                </button>
            ))}
        </div>
    );
}

function ReferencePicker({ items, selectedAnchorId, onSelectAnchor }) {
    return (
        <div className="panel p-5">
            <p className="kicker mb-1">01 — Reference</p>
            <h3 className="font-display text-2xl mb-3">Focus item</h3>

            {!items.length && (
                <div className="text-sm t-muted leading-relaxed">
                    Once suggestions appear, pick one piece to anchor your next request.
                </div>
            )}

            {items.length > 0 && (
                <div className="reference-list">
                    {items.map((item) => {
                        const aid = formatArticleId(item.article_id);
                        const isActive = aid === selectedAnchorId;
                        const title = item.product_type || item.title || "Item";
                        const subtitle = item.colour_group || item.subtitle || "";

                        return (
                            <button
                                key={aid}
                                className={"reference-item " + (isActive ? "active" : "")}
                                onClick={() => onSelectAnchor(aid)}
                            >
                                <img
                                    src={item.image_url || imagePathFromId(aid)}
                                    alt={title}
                                    className="reference-thumb"
                                />
                                <div className="reference-meta">
                                    <p className="title">{title}</p>
                                    <p className="subtitle">{subtitle || "No color specified"}</p>
                                    <p className="article">#{aid}</p>
                                </div>
                                <span className="reference-cta">Use</span>
                            </button>
                        );
                    })}
                </div>
            )}
        </div>
    );
}

function MoodBoardPanel() {
    return (
        <div className="panel p-5">
            <p className="kicker mb-1">02 — House notes</p>
            <h3 className="font-display text-2xl mb-3">How to style</h3>
            <ol className="notes-list">
                <li>Upload a reference image, or name a piece by its article id.</li>
                <li>Pick a focus item to anchor — pairings build around it.</li>
                <li>Ask for matching bottoms, outerwear, or a colour variant.</li>
            </ol>
        </div>
    );
}

function Composer({
    chatInput,
    setChatInput,
    currentImageBase64,
    onUploadClick,
    onFileChange,
    onRemoveImage,
    onSend,
    fileInputRef,
    textAreaRef,
    isLoading,
}) {
    return (
        <div className="composer">
            {currentImageBase64 && (
                <div className="flex items-center gap-3">
                    <img src={currentImageBase64} className="h-20 w-20 object-cover rounded border border-black/10" alt="Preview" />
                    <div>
                        <p className="text-sm t-muted">A new image resets the context and focus item.</p>
                        <button className="text-sm t-accent hover:underline mt-1" onClick={onRemoveImage}>Remove image</button>
                    </div>
                </div>
            )}

            <div className="composer-row">
                <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={onFileChange} />
                <button className="btn h-[52px] w-[52px] flex items-center justify-center" disabled={isLoading} onClick={onUploadClick} aria-label="Upload image">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                </button>
                <textarea
                    ref={textAreaRef}
                    className="composer-input"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    placeholder="Example: Find shoes that match this shirt"
                    onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                            e.preventDefault();
                            onSend();
                        }
                    }}
                />
                <button className="btn-primary h-[54px] px-8" disabled={isLoading} onClick={onSend}>
                    {isLoading ? "Sending…" : "Send"}
                </button>
            </div>
        </div>
    );
}

function Workspace({
    sessionId,
    messages,
    quickActions,
    selectedAnchorId,
    anchorItems,
    onSelectAnchor,
    onOpenItem,
    onPickQuickAction,
    onResetSession,
    onConfirmIntent,
    onSend,
    chatInput,
    setChatInput,
    currentImageBase64,
    onUploadClick,
    onFileChange,
    onRemoveImage,
    fileInputRef,
    textAreaRef,
    isLoading,
    chatBoxRef,
    latestMeta,
}) {
    return (
        <section className="workspace">
            <div className="layout">
                <div className="panel chat-panel">
                    <StatusRibbon meta={latestMeta} sessionId={sessionId} />

                    <div className="chat-header">
                        <div>
                            <p className="kicker mb-1">Styling session</p>
                            <h2 className="font-display text-4xl leading-none">The Atelier Desk</h2>
                        </div>
                        <div className="flex items-center gap-4">
                            <span className={"anchor-pill " + (!selectedAnchorId ? "hidden" : "")}>focus #{selectedAnchorId}</span>
                            <button className="btn px-4 py-2 text-xs" onClick={onResetSession}>Reset</button>
                        </div>
                    </div>

                    <QuickActions actions={quickActions} onPick={onPickQuickAction} />

                    <div className="chat-box" ref={chatBoxRef}>
                        {messages.map((message) => (
                            message.role === "user"
                                ? <UserMessage key={message.id} message={message} />
                                : <AiMessage key={message.id} message={message} onOpenItem={onOpenItem} onConfirmIntent={onConfirmIntent} />
                        ))}
                        {isLoading && <div className="typing">Generating outfit suggestions...</div>}
                    </div>

                    <Composer
                        chatInput={chatInput}
                        setChatInput={setChatInput}
                        currentImageBase64={currentImageBase64}
                        onUploadClick={onUploadClick}
                        onFileChange={onFileChange}
                        onRemoveImage={onRemoveImage}
                        onSend={onSend}
                        fileInputRef={fileInputRef}
                        textAreaRef={textAreaRef}
                        isLoading={isLoading}
                    />
                </div>

                <div className="side-panel">
                    <ReferencePicker items={anchorItems} selectedAnchorId={selectedAnchorId} onSelectAnchor={onSelectAnchor} />
                    <MoodBoardPanel />
                </div>
            </div>
        </section>
    );
}

function App() {
    const [started, setStarted] = useState(false);
    const [bootstrap, setBootstrap] = useState(DEFAULT_BOOTSTRAP);
    const [messages, setMessages] = useState([]);
    const [chatInput, setChatInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [currentImageBase64, setCurrentImageBase64] = useState(null);
    const [selectedAnchorId, setSelectedAnchorId] = useState("");
    const [anchorItems, setAnchorItems] = useState([]);
    const [quickActions, setQuickActions] = useState(DEFAULT_BOOTSTRAP.suggested_prompts.slice(0, 6));
    const [modalItems, setModalItems] = useState([]);
    const [modalIndex, setModalIndex] = useState(0);
    const [latestMeta, setLatestMeta] = useState(null);

    const chatBoxRef = useRef(null);
    const fileInputRef = useRef(null);
    const textAreaRef = useRef(null);
    const sessionIdRef = useRef(getSessionId());
    const lastUserImageRef = useRef(null);
    const prevMsgCountRef = useRef(0);

    const api = useMemo(() => window.MragApi || {}, []);

    useEffect(() => {
        let alive = true;
        const loader = api.getBootstrap
            ? api.getBootstrap()
            : fetch("/api/frontend/bootstrap").then((response) => response.json());

        Promise.resolve(loader)
            .then((data) => {
                if (!alive || !data) return;
                setBootstrap((prev) => ({ ...prev, ...data }));
                if (Array.isArray(data.suggested_prompts) && data.suggested_prompts.length > 0) {
                    setQuickActions(data.suggested_prompts.slice(0, 6));
                }
            })
            .catch(() => {
                // Keep default bootstrap values.
            });

        return () => {
            alive = false;
        };
    }, [api]);

    useEffect(() => {
        if (!started) return undefined;

        const html = document.documentElement;
        const body = document.body;
        const prevHtmlOverflow = html.style.overflow;
        const prevBodyOverflow = body.style.overflow;
        const prevHtmlHeight = html.style.height;
        const prevBodyHeight = body.style.height;

        html.style.overflow = "hidden";
        body.style.overflow = "hidden";
        html.style.height = "100%";
        body.style.height = "100%";
        window.scrollTo(0, 0);

        return () => {
            html.style.overflow = prevHtmlOverflow;
            body.style.overflow = prevBodyOverflow;
            html.style.height = prevHtmlHeight;
            body.style.height = prevBodyHeight;
        };
    }, [started]);

    useEffect(() => {
        if (!chatBoxRef.current) return;
        chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }, [messages, isLoading]);

    useEffect(() => {
        const grew = messages.length > prevMsgCountRef.current;
        prevMsgCountRef.current = messages.length;
        if (!grew || !window.gsap || !chatBoxRef.current) return;
        const nodes = chatBoxRef.current.querySelectorAll(".chat-message");
        const lastNode = nodes[nodes.length - 1];
        if (!lastNode) return;
        window.gsap.fromTo(
            lastNode,
            { opacity: 0, y: 16 },
            { opacity: 1, y: 0, duration: 0.45, ease: "power2.out" }
        );
    }, [messages]);

    function clearFocusState() {
        setSelectedAnchorId("");
        setAnchorItems([]);
    }

    function handleFileChange(event) {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            const result = e.target && e.target.result ? e.target.result : null;
            setCurrentImageBase64(result);
            clearFocusState();
        };
        reader.readAsDataURL(file);
    }

    function handleRemoveImage() {
        setCurrentImageBase64(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }

    async function handleResetSession() {
        const sid = sessionIdRef.current;
        try {
            if (api.resetSession) {
                await api.resetSession(sid);
            } else {
                await fetch("/api/session/reset", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sid }),
                });
            }
        } catch (_) {
            // Ignore reset errors and clear UI anyway.
        }

        setMessages([]);
        setChatInput("");
        setCurrentImageBase64(null);
        setLatestMeta(null);
        setModalItems([]);
        setModalIndex(0);
        clearFocusState();
        setQuickActions((bootstrap.suggested_prompts || DEFAULT_BOOTSTRAP.suggested_prompts).slice(0, 6));
    }

    const updateMessageById = (id, patch) => {
        if (!id) return;
        setMessages((prev) => prev.map((msg) => (
            msg.id === id ? { ...msg, ...patch } : msg
        )));
    };

    async function sendRequest({ text, image, confirmedIntent, userDisplayText }) {
        const payloadText = String(text || "").trim();
        const explicitImage = image === undefined ? null : image;
        const payloadImage = explicitImage || (confirmedIntent ? lastUserImageRef.current : null);
        if (!payloadText && !payloadImage) return;

        const displayText = userDisplayText || payloadText;
        setMessages((prev) => prev.concat([{ id: uid(), role: "user", text: displayText, image: payloadImage }]));

        if (!userDisplayText) {
            setChatInput("");
            handleRemoveImage();
        }

        if (payloadImage) {
            lastUserImageRef.current = payloadImage;
        }

        setIsLoading(true);
        let aiMessageId = "";

        try {
            const requestPayload = {
                text: payloadText,
                image: payloadImage,
                session_id: sessionIdRef.current,
                selected_anchor_id: selectedAnchorId || null,
                confirmed_intent: confirmedIntent || null,
                new_image_context: Boolean(payloadImage && !confirmedIntent),
                response_mode: "rich",
                include_debug: true,
                max_ui_items: 10,
                stream: true,
            };

            aiMessageId = uid();
            setMessages((prev) => prev.concat([
                {
                    id: aiMessageId,
                    role: "ai",
                    text: "",
                    intent: "",
                    intentInfo: null,
                    intentOptions: [],
                    intentQuery: "",
                    intentResolved: false,
                    items: [],
                    cards: [],
                    meta: null,
                    error: false,
                },
            ]));

            const updateAiMessage = (patch) => updateMessageById(aiMessageId, patch);

            const applyIntentPayload = (payload) => {
                const patch = {};
                if (payload && payload.intent) {
                    patch.intent = payload.intent;
                }
                if (payload && Array.isArray(payload.intent_options)) {
                    patch.intentOptions = payload.intent_options;
                    patch.intentResolved = false;
                }
                if (payload && payload.intent_query) {
                    patch.intentQuery = payload.intent_query;
                }
                if (Object.keys(patch).length > 0) {
                    updateAiMessage(patch);
                }
            };

            let streamedText = "";

            const appendDelta = (delta) => {
                if (!delta) return;
                streamedText += delta;
            };

            if (api.chatStream) {
                let streamError = null;
                let pendingItems = [];
                await api.chatStream(requestPayload, (eventName, data) => {
                    if (eventName === "meta") {
                        const normalizedItems = normalizeItems(data && data.items ? data.items : []);
                        pendingItems = normalizedItems;
                        applyIntentPayload(data);
                        return;
                    }
                    if (eventName === "delta") {
                        appendDelta(data && data.delta ? data.delta : "");
                        return;
                    }
                    if (eventName === "done") {
                        const finalText = data && data.message ? data.message : streamedText;
                        updateAiMessage({ text: finalText, items: pendingItems, cards: pendingItems });
                        applyIntentPayload(data);
                        if (pendingItems.length > 0) {
                            setAnchorItems(pendingItems.slice(0, 10));
                            setSelectedAnchorId(formatArticleId(pendingItems[0].article_id) || "");
                        } else {
                            clearFocusState();
                        }
                        streamedText = "";
                        pendingItems = [];
                        return;
                    }
                    if (eventName === "error") {
                        streamedText = "";
                        pendingItems = [];
                        streamError = data && data.error ? data.error : "Streaming error";
                    }
                });
                if (streamError) {
                    throw new Error(streamError);
                }
            } else {
                const data = api.chat
                    ? await api.chat({ ...requestPayload, stream: false })
                    : await fetch("/api/chat", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ ...requestPayload, stream: false }),
                    }).then((response) => response.json());

                const payload = data && data.data ? data.data : data;
                const normalizedItems = normalizeItems(payload && payload.items ? payload.items : []);
                updateAiMessage({
                    text: payload && payload.message ? payload.message : "",
                    items: normalizedItems,
                    cards: normalizedItems,
                    intent: payload && payload.intent ? payload.intent : "",
                    intentOptions: Array.isArray(payload && payload.intent_options) ? payload.intent_options : [],
                    intentQuery: payload && payload.intent_query ? payload.intent_query : "",
                    intentResolved: false,
                });
                if (normalizedItems.length > 0) {
                    setAnchorItems(normalizedItems.slice(0, 10));
                    setSelectedAnchorId(formatArticleId(normalizedItems[0].article_id) || "");
                } else {
                    clearFocusState();
                }
            }

            setLatestMeta(null);
            setQuickActions(quickActions);
        } catch (_) {
            if (aiMessageId) {
                updateMessageById(aiMessageId, {
                    text: "Cannot connect to the AI server.",
                    items: [],
                    cards: [],
                    meta: null,
                    error: true,
                });
            } else {
                setMessages((prev) => prev.concat([
                    {
                        id: uid(),
                        role: "ai",
                        text: "Cannot connect to the AI server.",
                        intent: "",
                        intentInfo: null,
                        items: [],
                        cards: [],
                        meta: null,
                        error: true,
                    },
                ]));
            }
        } finally {
            setIsLoading(false);
        }
    }

    async function sendMessage() {
        const payloadText = chatInput.trim();
        const payloadImage = currentImageBase64;
        if (!payloadText && !payloadImage) return;
        await sendRequest({ text: payloadText, image: payloadImage });
    }

    function handleConfirmIntent(message, intentId) {
        const intentInfo = INTENT_OPTIONS.find((opt) => opt.id === intentId);
        updateMessageById(message.id, { intentResolved: true });
        const queryText = message.intentQuery || message.text || "";
        sendRequest({
            text: queryText,
            confirmedIntent: intentId,
            userDisplayText: intentInfo ? `Intent confirmed: ${intentInfo.label}` : "Intent confirmed",
        });
    }

    return (
        <>
            {!started && <HeroScreen bootstrap={bootstrap} onStart={() => setStarted(true)} />}

            {started && (
                <Workspace
                    sessionId={sessionIdRef.current}
                    messages={messages}
                    quickActions={quickActions}
                    selectedAnchorId={selectedAnchorId}
                    anchorItems={anchorItems}
                    onSelectAnchor={(aid) => setSelectedAnchorId(formatArticleId(aid))}
                    onOpenItem={(item, items) => {
                        const normalized = normalizeItems(items || []);
                        const aid = formatArticleId(item.article_id);
                        const idx = Math.max(0, normalized.findIndex((it) => formatArticleId(it.article_id) === aid));
                        setModalItems(normalized);
                        setModalIndex(idx);
                    }}
                    onPickQuickAction={(text) => {
                        setChatInput(text);
                        if (textAreaRef.current) textAreaRef.current.focus();
                    }}
                    onResetSession={handleResetSession}
                        onConfirmIntent={handleConfirmIntent}
                    onSend={sendMessage}
                    chatInput={chatInput}
                    setChatInput={setChatInput}
                    currentImageBase64={currentImageBase64}
                    onUploadClick={() => fileInputRef.current && fileInputRef.current.click()}
                    onFileChange={handleFileChange}
                    onRemoveImage={handleRemoveImage}
                    fileInputRef={fileInputRef}
                    textAreaRef={textAreaRef}
                    isLoading={isLoading}
                    chatBoxRef={chatBoxRef}
                    latestMeta={latestMeta}
                />
            )}

            <ProductModal
                items={modalItems}
                index={modalIndex}
                onNavigate={(nextIndex) => setModalIndex(nextIndex)}
                onClose={() => {
                    setModalItems([]);
                    setModalIndex(0);
                }}
            />
        </>
    );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
