<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { onMount } from 'svelte';
	import {
		listConnections,
		createConnection,
		updateConnection,
		verifyConnection,
		deleteConnection,
		type Connection
	} from '$lib/apis/admin';
	import { refreshChatState } from '$lib/stores/chat';
	import { t } from '$lib/i18n';
	import ToggleSwitch from '$lib/components/common/ToggleSwitch.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';

	let loading = $state(true);
	let saving = $state(false);
	let verifying = $state(false);
	let verifyResult = $state<{ ok: boolean; message: string } | null>(null);

	// The single vertex connection this tab manages (if any).
	let conn = $state<Connection | null>(null);

	// Form state
	let enabled = $state(false);
	let authMode = $state<'vertex' | 'api_key'>('vertex');
	let project = $state('');
	let location = $state('global');
	let prefixId = $state('vertex');
	let models = $state('gemini-2.5-pro, gemini-2.5-flash');
	// Secret: JSON key (vertex) or API key. Blank means "keep existing".
	let secretInput = $state('');
	let hasSavedSecret = $state(false);

	onMount(load);

	async function load() {
		loading = true;
		try {
			const all = await listConnections();
			conn = all.find((c) => c.provider === 'vertex') ?? null;
			if (conn) {
				enabled = conn.enabled;
				authMode = (conn.data?.auth_mode as 'vertex' | 'api_key') || 'vertex';
				project = conn.data?.vertex_project || '';
				location = conn.data?.vertex_location || 'global';
				prefixId = conn.prefix_id || 'vertex';
				models = (conn.data?.models || []).join(', ');
				hasSavedSecret = !!conn.api_key;
			}
		} catch {
			toast.error($t('admin.vertex.loadError'));
		}
		loading = false;
	}

	function modelList(): string[] | undefined {
		const list = models
			.split(',')
			.map((m) => m.trim())
			.filter(Boolean);
		return list.length ? list : undefined;
	}

	async function save() {
		if (enabled && authMode === 'vertex' && !project.trim()) {
			toast.error($t('admin.vertex.projectRequired'));
			return;
		}
		saving = true;
		try {
			const payload = {
				name: 'Vertex',
				provider: 'vertex',
				prefix_id: prefixId.trim() || null,
				enabled,
				models: modelList(),
				auth_mode: authMode,
				vertex_project: project.trim(),
				vertex_location: location.trim() || 'global',
				// Only send the secret if the user entered a new value.
				...(secretInput.trim() ? { api_key: secretInput.trim() } : {})
			};

			if (conn) {
				await updateConnection(conn.id, payload);
			} else {
				await createConnection(payload);
			}
			toast.success($t('settings.saved'));
			secretInput = '';
			await load();
			refreshChatState();
		} catch (e) {
			toast.error(e instanceof Error ? e.message : $t('admin.failedToSave'));
		} finally {
			saving = false;
		}
	}

	async function verify() {
		if (!conn) {
			toast.error($t('admin.vertex.saveFirst'));
			return;
		}
		verifying = true;
		verifyResult = null;
		try {
			verifyResult = await verifyConnection(conn.id);
		} catch (e) {
			verifyResult = { ok: false, message: e instanceof Error ? e.message : 'Failed' };
		} finally {
			verifying = false;
		}
	}

	async function remove() {
		if (!conn) return;
		if (!confirm($t('admin.vertex.removeConfirm'))) return;
		try {
			await deleteConnection(conn.id);
			conn = null;
			enabled = false;
			hasSavedSecret = false;
			secretInput = '';
			toast.success($t('admin.vertex.removed'));
			refreshChatState();
		} catch {
			toast.error($t('admin.failedToSave'));
		}
	}

	const inputClass =
		'w-full mt-1 h-7 px-2 rounded-lg text-xs bg-gray-100 dark:bg-white/6 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-white/8 outline-none focus:border-blue-400 dark:focus:border-blue-500 transition-colors';
</script>

<div class="flex flex-col min-h-full">
	<h2 class="text-sm font-medium text-gray-900 dark:text-white mb-1">{$t('admin.vertex.title')}</h2>
	<p class="text-[11px] text-gray-400 dark:text-gray-600 mb-4">{$t('admin.vertex.subtitle')}</p>

	{#if loading}
		<div class="flex justify-center py-8"><Spinner size={16} /></div>
	{:else}
		<div class="flex flex-col gap-2.5">
			<!-- Master toggle -->
			<label class="flex items-center justify-between cursor-pointer">
				<span class="text-xs text-gray-600 dark:text-gray-400">{$t('admin.vertex.enable')}</span>
				<ToggleSwitch value={enabled} onchange={(v) => (enabled = v)} />
			</label>
			<p class="text-[11px] text-gray-400 dark:text-gray-600 -mt-1">
				{enabled ? $t('admin.vertex.enabledHint') : $t('admin.vertex.disabledHint')}
			</p>

			<!-- Auth mode -->
			<div class="flex items-center justify-between mt-2">
				<span class="text-xs text-gray-600 dark:text-gray-400">{$t('admin.vertex.authMode')}</span>
				<select
					bind:value={authMode}
					class="bg-transparent text-xs text-gray-600 dark:text-gray-400 outline-none cursor-pointer"
				>
					<option value="vertex">{$t('admin.vertex.authVertex')}</option>
					<option value="api_key">{$t('admin.vertex.authApiKey')}</option>
				</select>
			</div>

			{#if authMode === 'vertex'}
				<div>
					<label class="text-xs text-gray-600 dark:text-gray-400" for="v-project">{$t('admin.vertex.project')}</label>
					<input id="v-project" type="text" bind:value={project} placeholder="my-gcp-project" class={inputClass} />
				</div>
				<div>
					<label class="text-xs text-gray-600 dark:text-gray-400" for="v-location">{$t('admin.vertex.location')}</label>
					<input id="v-location" type="text" bind:value={location} placeholder="global" class={inputClass} />
					<p class="text-[11px] text-gray-400 dark:text-gray-600 mt-0.5">{$t('admin.vertex.locationHint')}</p>
				</div>
				<div>
					<label class="text-xs text-gray-600 dark:text-gray-400" for="v-creds">{$t('admin.vertex.credentials')}</label>
					<textarea
						id="v-creds"
						bind:value={secretInput}
						rows="4"
						placeholder={hasSavedSecret ? $t('admin.vertex.secretSaved') : '{ "type": "service_account", ... }'}
						class="{inputClass} h-auto py-1.5 font-mono resize-y"
					></textarea>
					<p class="text-[11px] text-gray-400 dark:text-gray-600 mt-0.5">{$t('admin.vertex.credentialsHint')}</p>
				</div>
			{:else}
				<div>
					<label class="text-xs text-gray-600 dark:text-gray-400" for="v-apikey">{$t('admin.vertex.apiKey')}</label>
					<input
						id="v-apikey"
						type="password"
						bind:value={secretInput}
						autocomplete="new-password"
						placeholder={hasSavedSecret ? $t('admin.vertex.secretSaved') : 'AIza...'}
						class="{inputClass} font-mono"
					/>
					<p class="text-[11px] text-gray-400 dark:text-gray-600 mt-0.5">{$t('admin.vertex.apiKeyHint')}</p>
				</div>
			{/if}

			<!-- Models -->
			<div class="mt-2">
				<label class="text-xs text-gray-600 dark:text-gray-400" for="v-models">{$t('admin.vertex.models')}</label>
				<input id="v-models" type="text" bind:value={models} placeholder="gemini-2.5-pro, gemini-2.5-flash" class="{inputClass} font-mono" />
				<p class="text-[11px] text-gray-400 dark:text-gray-600 mt-0.5">{$t('admin.vertex.modelsHint')}</p>
			</div>

			<!-- Prefix -->
			<div>
				<label class="text-xs text-gray-600 dark:text-gray-400" for="v-prefix">{$t('admin.vertex.prefix')}</label>
				<input id="v-prefix" type="text" bind:value={prefixId} placeholder="vertex" class="{inputClass} font-mono" />
				<p class="text-[11px] text-gray-400 dark:text-gray-600 mt-0.5">{$t('admin.vertex.prefixHint')}</p>
			</div>

			<!-- Verify -->
			{#if conn}
				<div class="flex items-center gap-2 mt-1">
					<button
						class="h-7 px-2.5 rounded-lg text-xs bg-gray-200/50 dark:bg-white/8 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors disabled:opacity-50"
						onclick={verify}
						disabled={verifying}
					>{verifying ? '...' : $t('admin.vertex.verify')}</button>
					{#if verifyResult}
						<span class="text-[11px] {verifyResult.ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-red-500'}">
							{verifyResult.message}
						</span>
					{/if}
				</div>
			{/if}
		</div>

		<!-- Actions -->
		<div class="mt-auto pt-6 flex justify-between items-center">
			{#if conn}
				<button
					class="text-[12px] text-red-400 hover:text-red-500 transition-colors"
					onclick={remove}
				>{$t('admin.vertex.remove')}</button>
			{:else}
				<span></span>
			{/if}
			<button
				class="text-[13px] text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors duration-100 disabled:opacity-50"
				onclick={save}
				disabled={saving}
			>{saving ? $t('admin.vertex.saving') : $t('settings.save')}</button>
		</div>
	{/if}
</div>
